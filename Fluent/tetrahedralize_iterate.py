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
import pyvista as pv
import custom_vmtkflowextensions
import vtk
import numpy as np
import meshio
root_dir = str(sys.argv[1])
target = int(sys.argv[2])
start, end = int(sys.argv[3]), int(sys.argv[4])

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)
    
    
def tetrahedralize(edge, name, idir, tdir, odir):
    arg = (
                f" vmtkmeshgenerator -ifile {idir+'/'+ name}.vtk -edgelength {edge} "
                f" -boundarylayer 0 "
                f" -boundarylayeroncaps 1 -tetrahedralize 1 -ofile {tdir+'/'+name}.vtu"
            )
    os.system(arg)
    mesh = meshio.read(f'{tdir}/{name}.vtu')
    temp_Npoints = len(mesh.points)
    print(temp_Npoints)    
    return temp_Npoints
    


def cfd_mesher(root_dir, name, idir, odir):
    '''2'''
    
    tdir = root_dir + f'/volume_meshes_temp_{target}'
    
    patience = -1
    if not os.path.exists(odir):
        os.makedirs(odir)
    if not os.path.exists(tdir):
        os.makedirs(tdir)
        
    all_points = []
    if (not os.path.exists(f'{odir}/{name}.vtu')) and (not os.path.exists(f'{odir}/{name}_{target}.vtu')):
        
        patience = 0
        guess1 = 2.1605
        guess2 = 2.1605

        # Define the tolerance for your condition (adjust this as needed)
        tolerance = 0#target# * 0.005
        points1 = tetrahedralize(guess1, name, idir, tdir, odir)         
        
        all_points.append(points1)
        if points1 < target:
            guess1 = guess1 - 0.8
            points1 = tetrahedralize(guess1, name, idir, tdir, odir)
            all_points.append(points1)
        print(all_points)
        points2 = tetrahedralize(guess2, name, idir, tdir, odir)
        all_points.append(points2)
        if points2 > target:
            guess2 = guess1 + 1
            points2 = tetrahedralize(guess2, name, idir, tdir, odir)
            all_points.append(points2)
        print(all_points)
        # Define the initial difference from the target for each guess
        diff1 = abs(points1 - target)
        diff2 = abs(points2 - target)
        
        
        current_guess = (guess1 + guess2) / 2
        best_difference = 100000
        best_guess = current_guess
        print(all_points)
        while diff1 > tolerance and diff2 > tolerance and patience < 15:
            current_guess = (guess1 + guess2) / 2
            current_Npoints = tetrahedralize(current_guess, name, idir, tdir, odir)
            all_points.append(current_Npoints)
            current_difference = abs(current_Npoints - target)
            if current_difference == best_difference:
                break
            if  current_difference <best_difference:
                best_guess = current_guess
                best_difference = current_difference       
            if current_Npoints > target:
                guess1 = current_guess
                diff1 = abs(current_Npoints - target)
            elif current_Npoints < target:
                guess2 = current_guess
                diff2 = abs(current_Npoints - target)
            else:
                break
            print(all_points)
            patience+=1
        tetrahedralize(best_guess, name, idir, tdir, odir)
        pv.read(f"{tdir+'/'+name}.vtu").save(f"{odir+'/'+name}.vtu") 
        
        
def id_fix(root_dir, name, odir):
    '''3'''
    idir = odir
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
        
        
def process(root_dir):
    names = [os.path.splitext(name)[0].split('/')[-1] for name in glob.glob(f'{root_dir}/surface_meshes/*')]
    names = natural_sort(names)    
    
    odir = root_dir + f'/volume_meshes_{target}/generated_samples_unregistered'
    idir = root_dir + '/surface_meshes'
    for name in names[start:end]:
        print(name)
        print("1. Volume Meshing")
        cfd_mesher(root_dir, name, idir, odir)
        id_fix(root_dir, name, odir)
    
    
if __name__=="__main__":
    process(root_dir)
