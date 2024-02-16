import os, sys, copy
import vmtk
from vmtk import vmtkscripts
import glob
import sys
import re
import numpy as np
import vtk
from vtk.vtkCommonCore import vtkObject
from vtk.util import numpy_support as ns
import meshio
import custom_vmtkflowextensions
import pyvista as pv
root_dir = str(sys.argv[1])
start = int(sys.argv[2]) if len(sys.argv) >= 3 else 0
end = int(sys.argv[3]) if len(sys.argv) >= 4 else 10000


def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

    
def calculate_new_edge_length(new_volume, old_edge_length = 1.02, old_volume = 14300):
    ratio = (new_volume / old_volume) ** (1/3)
    new_edge_length = old_edge_length * ratio
    return new_edge_length

def create_extension(root_dir, name):
    '''1'''

    odir = root_dir + '/extensions'
    idir = root_dir + '/generated_samples_clipped'
    if not os.path.exists(odir):
        os.makedirs(odir)
        
    if not os.path.exists(f"{odir+'/'+name}.vtk"): 
        # read surface
        surf_reader = vmtk.vmtksurfacereader.vmtkSurfaceReader()
        surf_reader.InputFileName = f'{idir}/{name}.vtk'
        surf_reader.Execute()
        
        # extensions
        extender = vmtk.vmtkflowextensions.vmtkFlowExtensions()
        extender.Surface = surf_reader.Surface
        extender.ExtensionMode = "boundarynormal"
        extender.Interactive = 0
        extender.ExtensionLength = 50
        extender.TargetNumberOfBoundaryPoints = 50
        extender.Execute()

        # recompute normals
        normals = vmtk.vmtksurfacenormals.vmtkSurfaceNormals()
        normals.Surface = extender.Surface
        normals.Execute()

        # Update the surface for the next extension
        surf_reader.Surface = normals.Surface

        # write surface
        writer = vmtk.vmtksurfacewriter.vmtkSurfaceWriter()
        writer.Surface = normals.Surface
        writer.OutputFileName = f'{odir}/{name}.vtk'
        writer.Execute()
    

def tetrahedralize(edge, name, idir, odir):
    arg = (
                f" vmtkmeshgenerator -ifile {idir+'/'+ name}.vtk -edgelength {edge} "
                f" -boundarylayer 0"
                f" -boundarylayeroncaps 1 -tetrahedralize 1 -ofile {odir+'/'+name}.vtu"
            )
    os.system(arg)
    mesh = meshio.read(f'{odir}/{name}.vtu')
    Npoints = len(mesh.cells[0].data)
    print(Npoints)
    return Npoints
    


def cfd_mesher(root_dir, name):
    '''2'''
    tdir = root_dir + '/volume_meshes_temp'
    odir = root_dir + '/volume_vtu'
    idir = root_dir + '/extensions'
    patience = -1
    if not os.path.exists(odir):
        os.makedirs(odir)
    if not os.path.exists(tdir):
        os.makedirs(tdir)
    if not os.path.exists(f'{odir}/{name}.vtu'):
        patience = 0
        volume = pv.read(f'{idir}/{name}.vtk').volume
        
        guess = calculate_new_edge_length(volume)
        points = tetrahedralize(guess, name, idir, odir)

        

def id_fix(root_dir, name):
    '''3'''
    odir = root_dir + '/volume_vtu'
    idir = root_dir + '/volume_vtu'
    
    if not os.path.exists(odir):
        os.makedirs(odir)
        
    if os.path.exists(f"{idir+'/'+name}.vtu"): 
        print('id fixing')
        # read vtu
        mesh_reader = vmtk.vmtkmeshreader.vmtkMeshReader()
        mesh_reader.InputFileName = f"{idir}/{name}.vtu"
        mesh_reader.Execute()
        # convert vtu to np array
        mesh2np = vmtk.vmtkmeshtonumpy.vmtkMeshToNumpy()
        mesh2np.Mesh = mesh_reader.Mesh
        mesh2np.Execute()
        mesh_arr = mesh2np.ArrayDict
        vtk_reader = vtk.vtkXMLUnstructuredGridReader()
        vtk_reader.SetFileName(f"{idir}/{name}.vtu")
        vtk_reader.Update()
        ugrid = vtk_reader.GetOutput()

        # get indices of two triangles (id=2, id=3)
        point4 = np.array([48.14547, 99.8016, 98.834335])
        point5 = np.array([64.640594, 85.095726, 27.556936])
        point6 = np.array([74.74099, 30.692032, 88.21257])

        cell_id2 = np.where(mesh_arr['CellData']['CellEntityIds']==2)[0][0]
        cell_id3 = np.where(mesh_arr['CellData']['CellEntityIds']==3)[0][0]
        cell_id4 = np.where(mesh_arr['CellData']['CellEntityIds']==4)[0][0]


        # get triangle verts
        points_id2 = copy.deepcopy(ns.vtk_to_numpy(ugrid.GetCell(cell_id2).GetPoints().GetData()))
        points_id3 = copy.deepcopy(ns.vtk_to_numpy(ugrid.GetCell(cell_id3).GetPoints().GetData()))
        points_id4 = copy.deepcopy(ns.vtk_to_numpy(ugrid.GetCell(cell_id4).GetPoints().GetData()))
        
        points = [point4, point5, point6]
        dists2 = []
        dists3 = []
        dists4 = []
        for point in points:
            dist_outlet_id2 = np.linalg.norm(point - points_id2[0])
            dist_outlet_id3 = np.linalg.norm(point - points_id3[0])
            dist_outlet_id4 = np.linalg.norm(point - points_id4[0])
            dists2.append(dist_outlet_id2)
            dists3.append(dist_outlet_id3)
            dists4.append(dist_outlet_id4)
        point2_newid = np.argmin(dists2) + 2
        point3_newid = np.argmin(dists3) + 2
        point4_newid = np.argmin(dists4) + 2

        for i, cell_id in enumerate(mesh_arr['CellData']['CellEntityIds']):
            if cell_id == 2:
                mesh_arr['CellData']['CellEntityIds'][i] = point2_newid
            elif cell_id == 3:
                mesh_arr['CellData']['CellEntityIds'][i] = point3_newid
            elif cell_id == 4:
                mesh_arr['CellData']['CellEntityIds'][i] = point4_newid

        # convert new np array to mesh
        np2mesh = vmtk.vmtknumpytomesh.vmtkNumpyToMesh()
        np2mesh.ArrayDict = mesh_arr
        np2mesh.Execute()
        # write mesh as vtu file
        writer = vmtk.vmtkmeshwriter.vmtkMeshWriter()
        writer.Mesh = np2mesh.Mesh
        writer.OutputFileName = f"{odir}/{name}.vtu"
        writer.Execute()
    
def msh_to_vtu(root_dir, name):
    '''4'''
    odir = root_dir + '/volume_msh'
    idir = root_dir + '/volume_vtu'

    if not os.path.exists(odir):
        os.makedirs(odir)
    if os.path.exists(f"{idir+'/'+name}.vtu"): 
        # read vtu
        mesh_reader = vmtk.vmtkmeshreader.vmtkMeshReader()
        mesh_reader.InputFileName = f"{idir}/{name}.vtu"
        mesh_reader.Execute()
        # write mesh as fluent file
        writer = vmtk.vmtkmeshwriter.vmtkMeshWriter()
        writer.Mesh = mesh_reader.Mesh
        writer.OutputFileName = f"{odir}/{name}.msh"
        writer.Execute()

def cfd_prepare(root_dir):

    # Get the directory where the Python script is located
    script_directory = os.path.dirname(os.path.abspath(__file__))
    code_dir = script_directory.split('/')[-1]
    jou = open(f'{code_dir}/autofluent_template.jou', 'r')
    lines = jou.readlines()
    jou.close()
    
    names = [os.path.splitext(name)[0].split('/')[-1] for name in glob.glob(f'{root_dir}/volume_msh/*.msh')]
    remove_names = [os.path.splitext(name)[0].split('/')[-1] for name in glob.glob(f'{root_dir}/cfd_csv/*.csv')]
    names = sorted(set(names).difference(set(remove_names)))

    for i in range(0, len(lines)):
        if 'root_folder/' in lines[i]:
            lines[i] = lines[i].replace('root_folder/',root_dir + '/')
    
        if '(list' in lines[i]:
            for j, name in enumerate(names):
                lines.insert(i+j+1, ' "' + name + '"\n')
            break

    jou = open(root_dir + '/autofluent.jou', 'w')
    jou.writelines(lines)
    jou.close()
    print('journal file written')

    pview = open(f'{code_dir}/pview_resample_cfd2vtu_template.py', 'r')

    lines = pview.readlines()
    for i in range(len(lines)):
        if 'D:/Fluent/' in lines[i]:
            lines[i] = lines[i].replace('root_folder/',root_dir + '/')

    pview = open(root_dir + '/pview_resample_cfd2vtu.py', 'w')
    pview.writelines(lines)
    pview.close()
    print('python file written')
        
        

def process(root_dir):
    names = [os.path.splitext(name)[0].split('/')[-1] for name in glob.glob(f'{root_dir}/generated_samples_clipped/*')]
    names = natural_sort(names)
    
    cfd_prepare(root_dir)
    
    for name in names[start:end]:
        try:
            print(name)
            if not os.path.exists(f'{root_dir}/volume_msh/{name}.msh'):
                print("1. Creating Extension")
                create_extension(root_dir, name)
                print("2. Volume Meshing")
                cfd_mesher(root_dir, name)
            #if os.path.exists(f'{root_dir}/volume_vtu/{name}.vtu'):
                print("3. ID fixing")
                id_fix(root_dir, name)
                print("4. Convert to msh")
                msh_to_vtu(root_dir, name)
                cfd_prepare(root_dir)
        except Exception as e:
            print(f"Error {e}, with {name}")
    cfd_prepare(root_dir)
    
if __name__=="__main__":
    process(root_dir)







