from vtk import *

quadric = vtkQuadric()
quadric.SetCoefficients(.5, 1, .2, 0, .1, 0, 0, .2, 0, 0)

sample = vtkSampleFunction()
sample.SetSampleDimensions(50,50,50)
sample.SetImplicitFunction(quadric)

contour = vtkContourFilter()
contour.SetInputConnection(sample.GetOutputPort())
contour.GenerateValues(5,0,1)

contourMapper = vtkPolyDataMapper()
contourMapper.SetInputConnection(contour.GetOutputPort())
contourMapper.SetScalarRange(0,1.2)

contourActor = vtkActor()
contourActor.SetMapper(contourMapper)

outline = vtkOutlineFilter()
outline.SetInputConnection(sample.GetOutputPort())

outlineMapper = vtkPolyDataMapper()
outlineMapper.SetInputConnection(outline.GetOutputPort())

outlineActor = vtkActor()
outlineActor.SetMapper(outlineMapper)
outlineActor.GetProperty().SetColor(1,1,1)

ren = vtkRenderer()
ren.SetBackground(0.188,0.373,0.647)
ren.AddActor(contourActor)
ren.AddActor(outlineActor)

renWin = vtkRenderWindow()
renWin.AddRenderer(ren)
renWin.SetWindowName("IsoSurface")
renWin.SetSize(500,500)

iren = vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

renWin.Render()
iren.Initialize()
iren.Start()