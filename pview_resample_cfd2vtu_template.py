from genericpath import isdir
from paraview.simple import *
# note install pv 5.8+ .tar.gz for import vtk to work:
from paraview import numpy_support as ns
from paraview import servermanager as sm
import os, sys, vtk
import numpy as np
import re
import glob

# Streamlines
# Table to Points
# Point Volume Interpolator
# Merge Vector Components x,y,z velocity -> Velocity
# Resample with Dataset
# Stream Tracer Seed type Point Cloud -> large than the object, Number of points = 10000



def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def resample_cfd2vtu(root_folder):
    '''
    - For each case in directory:
        - load cfd .csv and deformed .vtu
        - resample dataset (source is .csv), target is vtu
        - save out file as csv (ideally)
    '''
    idir = f"{root_folder}/cfd_csv" # cfd folder
    vdir = f"{root_folder}/generated_samples" # volume mesh folder
    odir = f"{root_folder}/cfd_vtu" # result folder
    sdir = f"{root_folder}/streamlines"
    
    if not os.path.exists(odir):
        os.makedirs(odir)
    if not os.path.exists(sdir):
        os.makedirs(sdir)
    names =[name.replace('\\','/').split('/')[-1].replace('.csv','') for name in  glob.glob(f'{idir}/*.csv')]
    print(names)
    
    for name in names:
        if not os.path.exists(f"{odir+'/'+name}.vtu"): 
            # read .csv
            cfd_csv = CSVReader(registrationName='cfd_csv', FileName = f"{idir}/{name}.csv")
            UpdatePipeline(time=0.0, proxy=cfd_csv)

            # read target mesh .vtu
            print("Loading " + name)
            vtu = XMLUnstructuredGridReader(registrationName='vtu', FileName = [f"{vdir}/{name}.vtu"])
            UpdatePipeline(time=0.0, proxy=vtu)
            
            # csv to point cloud
            tableToPoints = TableToPoints(registrationName='tableToPoints', Input=cfd_csv)
            tableToPoints.XColumn = '    x-coordinate'
            tableToPoints.YColumn = '    y-coordinate'
            tableToPoints.ZColumn = '    z-coordinate'
            UpdatePipeline(time=0.0, proxy=tableToPoints)

            # create a new 'Merge Vector Components'
            mergeVectorComponents = MergeVectorComponents(registrationName='mergeVectorComponents', Input=tableToPoints)
            mergeVectorComponents.XArray = '      x-velocity'
            mergeVectorComponents.YArray = '      y-velocity'
            mergeVectorComponents.ZArray = '      z-velocity'
            mergeVectorComponents.OutputVectorName = 'velocity-xyz'

            UpdatePipeline(time=0.0, proxy=mergeVectorComponents)

            pointVolumeInterpolator = PointVolumeInterpolator(registrationName='PointVolumeInterpolator', Input=mergeVectorComponents, Source='Bounded Volume')
            pointVolumeInterpolator.Kernel = 'VoronoiKernel'
            pointVolumeInterpolator.Locator = 'Static Point Locator'
            pointVolumeInterpolator.Source.Padding = 5
            UpdatePipeline(time=0.0, proxy=pointVolumeInterpolator)
            
            # resample filter
            resampleWithDataset = ResampleWithDataset(registrationName='resampleWithDataset', 
                                                        SourceDataArrays=pointVolumeInterpolator,
                                                        DestinationMesh=vtu)
            UpdatePipeline(time=0.0, proxy=resampleWithDataset)
                        

            print("writing interpolated cfd data: " + name)
            print("")
            SaveData(f"{odir}/{name}.vtu", proxy=resampleWithDataset, PointDataArrays = ['        pressure', 'velocity-magnitude'])
            Delete(resampleWithDataset)
            Delete(pointVolumeInterpolator)
            Delete(tableToPoints)
            Delete(cfd_csv)
            Delete(vtu)
            Delete(mergeVectorComponents)
            del(resampleWithDataset)
            del(pointVolumeInterpolator)
            del(tableToPoints)
            del(cfd_csv)
            del(vtu)
            del(mergeVectorComponents)
                   
        if not os.path.exists(f"{sdir}/{name}.vtk"): 
            cfd_vtu = XMLUnstructuredGridReader(registrationName='cfd_vtu', FileName = [f"{odir}/{name}.vtu"])
            cfd_vtu.PointArrayStatus = ['         cell-id', '        pressure', '      x-velocity', '      y-velocity', '      z-velocity', 'nodenumber', 'velocity-magnitude', 'velocity-xyz', 'vtkValidPointMask']
            UpdatePipeline(time=0.0, proxy=cfd_vtu)

            streamTracer = StreamTracer(registrationName='StreamTracer1', Input=cfd_vtu, SeedType='Line')
            streamTracer.Vectors = ['POINTS', 'velocity-xyz']
            streamTracer.MaximumStreamlineLength = 97.75109672546387

            streamTracer.SeedType.Point1 = [35.488807678222656, 20.36954689025879, 15.263551712036133]
            streamTracer.SeedType.Point2 = [89.19976806640625, 106.39297485351562, 113.0146484375]
            UpdatePipeline(time=0.0, proxy=streamTracer)

            streamTracer.SeedType = 'Point Cloud'
            streamTracer.SeedType.Center = [63.5, 63.5, 63.5]
            streamTracer.SeedType.Radius = 100
            streamTracer.SeedType.NumberOfPoints = 20000  # Change 100 to your desired number
            UpdatePipeline(time=0.0, proxy=streamTracer)
            
            SaveData(f"{sdir}/{name}.vtk", proxy=streamTracer, PointDataArrays=['         cell-id', '        pressure', '      x-velocity', '      y-velocity', '      z-velocity', 'velocity-xyz', 'AngularVelocity', 'IntegrationTime', 'Normals', 'Rotation', 'Vector', 'Vorticity', 'nodenumber', 'velocity-magnitude', 'vtkValidPointMask'],
    CellDataArrays=['ReasonForTermination', 'SeedIds'])

            Delete(cfd_vtu)            
            Delete(streamTracer)
           
            del(cfd_vtu)
            del(streamTracer)

root_folder = f'F:/Fluent/root_folder/'
resample_cfd2vtu(root_folder)
