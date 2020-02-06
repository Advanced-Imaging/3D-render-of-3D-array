# import SimpleITK as sitk
import vtk
import numpy as np
from utils_func import make3DGaussianKernel
from vtk.util.vtkConstants import *


def numpy2VTK(img, spacing=[1.0, 1.0, 1.0],margin=0):
    # evolved from code from Stou S.,
    # on http://www.siafoo.net/snippet/314
    importer = vtk.vtkImageImport()

    img_data = img.astype('uint8')
    img_string = img_data.tostring()  # type short
    dim = img.shape

    importer.CopyImportVoidPointer(img_string, len(img_string))
    importer.SetDataScalarType(VTK_UNSIGNED_CHAR)
    importer.SetNumberOfScalarComponents(1)

    extent = importer.GetDataExtent()
    importer.SetDataExtent(extent[0], extent[0] + dim[2] - 1,
                           extent[2], extent[2] + dim[1] - 1,
                           extent[4], extent[4] + dim[0] - 1)
    importer.SetWholeExtent(extent[0], extent[0] + dim[2] - 1 + margin,
                            extent[2], extent[2] + dim[1] - 1 + margin,
                            extent[4], extent[4] + dim[0] - 1 + margin)

    importer.SetDataSpacing(spacing[0], spacing[1], spacing[2])
    importer.SetDataOrigin(0, 0, 0)

    return importer


def volumeRender(img, tf=[], spacing=[1.0, 1.0, 1.0]):
    importer = numpy2VTK(img, spacing)

    # Transfer Functions
    opacity_tf = vtk.vtkPiecewiseFunction()
    color_tf = vtk.vtkColorTransferFunction()

    if len(tf) == 0:
        tf.append([img.min(), 0, 0, 0, 0])
        tf.append([img.max(), 1, 1, 1, 1])

    for p in tf:
        color_tf.AddRGBPoint(p[0], p[1], p[2], p[3])
        opacity_tf.AddPoint(p[0], p[4])

    # working on the GPU
    volMapper = vtk.vtkGPUVolumeRayCastMapper()
    volMapper.SetInputConnection(importer.GetOutputPort())

    # The property describes how the data will look
    volProperty = vtk.vtkVolumeProperty()
    volProperty.SetColor(color_tf)
    volProperty.SetScalarOpacity(opacity_tf)
    volProperty.ShadeOn()
    volProperty.SetInterpolationTypeToLinear()

    # working on the CPU
    # volMapper = vtk.vtkVolumeRayCastMapper()
    # compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()
    # compositeFunction.SetCompositeMethodToInterpolateFirst()
    # volMapper.SetVolumeRayCastFunction(compositeFunction)
    # volMapper.SetInputConnection(importer.GetOutputPort())

    # The property describes how the data will look
    volProperty = vtk.vtkVolumeProperty()
    volProperty.SetColor(color_tf)
    volProperty.SetScalarOpacity(opacity_tf)
    volProperty.ShadeOn()
    volProperty.SetInterpolationTypeToLinear()

    # Do the lines below speed things up?
    # pix_diag = 5.0
    # volMapper.SetSampleDistance(pix_diag / 5.0)
    # volProperty.SetScalarOpacityUnitDistance(pix_diag)

    vol = vtk.vtkVolume()
    vol.SetMapper(volMapper)
    vol.SetProperty(volProperty)

    return [vol]

def bounding_box(vtkRenderer,actors):
    volSource=actors[0]
    cubeAxesActor = vtk.vtkCubeAxesActor()
    cubeAxesActor.SetBounds(volSource.GetOutput().GetBounds())
    cubeAxesActor.SetCamera(vtkRenderer.GetActiveCamera())
    cubeAxesActor.GetTitleTextProperty(0).SetColor(1.0, 0.0, 0.0)
    cubeAxesActor.GetLabelTextProperty(0).SetColor(1.0, 0.0, 0.0)

    cubeAxesActor.GetTitleTextProperty(1).SetColor(0.0, 1.0, 0.0)
    cubeAxesActor.GetLabelTextProperty(1).SetColor(0.0, 1.0, 0.0)

    cubeAxesActor.GetTitleTextProperty(2).SetColor(0.0, 0.0, 1.0)
    cubeAxesActor.GetLabelTextProperty(2).SetColor(0.0, 0.0, 1.0)

    cubeAxesActor.DrawXGridlinesOn()
    cubeAxesActor.DrawYGridlinesOn()
    cubeAxesActor.DrawZGridlinesOn()
    if vtk.VTK_MAJOR_VERSION > 5:
        cubeAxesActor.SetGridLineLocation(vtk.VTK_GRID_LINES_FURTHEST)
    cubeAxesActor.XAxisMinorTickVisibilityOff()
    cubeAxesActor.YAxisMinorTickVisibilityOff()
    cubeAxesActor.ZAxisMinorTickVisibilityOff()
    actors.append(cubeAxesActor)

    return actors

def vtk_basic(actors):
    """
    Create a window, renderer, interactor, add the actors and start the thing

    Parameters
    ----------
    actors :  list of vtkActors

    Returns
    -------
    nothing
    """

    # create a rendering window and renderer
    ren = vtk.vtkRenderer()
    #actors=bounding_box(ren,actors)
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetSize(600, 600)
    #ren.SetBackground( 0.1, 0.2, 0.3)

    # create a renderwindowinteractor
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    for a in actors:
        # assign actor to the renderer
        ren.AddActor(a)

    # render
    renWin.Render()

    # enable user interface interactor
    iren.Initialize()
    iren.Start()


#####
#####
from scipy.ndimage import convolve as convolve3d
#img = sitk.ReadImage( filename ) # SimpleITK object
#data = sitk.GetArrayFromImage( img ).astype('float') # numpy array
p1=0.999999 # probability of choice
vol_shape=(100,100,100)
#data=np.random.choice([0, 1], size=(100,100,200), p=[p1, 1-p1])
#data=np.random.choice([False, True], size=vol_shape, p=[p1, 1-p1])
point_location=(np.random.rand(3)*np.asarray(vol_shape)).astype('int')
data=np.zeros(shape=vol_shape,dtype='bool')
#data[tuple(point_location)]=True
data[vol_shape[0]//2,vol_shape[1]//2,vol_shape[2]//2]=True
replica=[]
rolls=(-4,4,-20)
axes=(0,1,2)
for i in range(len(rolls)):
    for j in range(len(rolls)):
        replica.append(np.roll(data,rolls[i],axis=axes[j]))
for i in range(len(replica)):
    data+=replica[i]
data=data.astype('float64')
from scipy.stats.mstats import mquantiles
PSF=True
if PSF:
    kernel=make3DGaussianKernel([50,50,50],[3,3,9])
    data=convolve3d(data,kernel,mode='constant')
    q = mquantiles(data.flatten(),[0.7,0.95,0.98,0.99,0.999])
    for i in range(len(q)):
        q[i]=max(q[i],1)
    tf=[[0,0,0,0,0],[q[0],0,0,0,0],[q[1],0.4,0.4,0.4,0.1],[q[2],0.7,0.7,0.7,0.3],
        [q[3],0.8,0.8,0.8,0.4],[q[4],0.9,0.9,0.9,0.5],[data.max(),1,1,1,1]]
else:
    q = mquantiles(data.flatten(), [0.7, 0.98])
    q[0] = max(q[0], 1)
    q[1] = max(q[1], 1)
    tf = [[0, 0, 0, 0, 0], [q[0], 0, 0, 0, 0], [q[1], 1, 0.6, 0.6, 0.5],          [data.max(), 1, 1, 1, 1]]
data *= 255 / data.max()
actor_list = volumeRender(data, tf=tf)

vtk_basic(actor_list)

