import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys
import vtk
from PyQt5 import QtCore, QtGui, QtWidgets
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from Mainwindow_big import Ui_MainWindow
from test_pred import show_partseg


# import pcl.pcl_visualization


class Mywindow(QtWidgets.QMainWindow, Ui_MainWindow):

    def __init__(self):
        super(Mywindow, self).__init__()
        self.setupUi(self)
        self.setWindowTitle('毕业答辩成果演示')
        self.frame = QtWidgets.QFrame()
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        self.formLayout.addWidget(self.vtkWidget)
        self.ren = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)
        self.ren.SetBackground(1, 1, 1)  # 设置为白色
        self.vtkWidget.GetRenderWindow().Render()
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()
        # Create source
        self.pushButton_2.clicked.connect(self.last)
        self.pushButton_3.clicked.connect(self.next)
        self.pushButton.clicked.connect(self.pred)
        self.data_set=show_partseg(3)
        self.index=0
        self.display(self.index)
    def display(self,index):
        pcd = self.data_set.get_index(index)
        points = np.asarray(pcd.points)
        colors=np.asarray(pcd.colors)
        # 创建VTK的vtkPoints对象
        vtk_colors = vtk.vtkUnsignedCharArray()
        vtk_colors.SetNumberOfComponents(3)
        vtk_colors.SetName("Colors")
        for color in colors:
            vtk_colors.InsertNextTuple3(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
        vtk_points = vtk.vtkPoints()
        # 将NumPy数组中的点添加到vtkPoints对象中
        for point in points:
            vtk_points.InsertNextPoint(point[0], point[1], point[2])
        polydata = vtk.vtkPolyData()
        if colors.size > 0:
            polydata.GetPointData().SetScalars(vtk_colors)
        polydata.SetPoints(vtk_points)


        self.vtkWidget.GetRenderWindow().Render()

        glyphFilter = vtk.vtkVertexGlyphFilter()
        glyphFilter.SetInputData(polydata)
        glyphFilter.Update()
        dataMapper = vtk.vtkPolyDataMapper()
        dataMapper.SetInputConnection(glyphFilter.GetOutputPort())
        # Create an actor
        actor = vtk.vtkActor()
        actor.SetMapper(dataMapper)
        actor1 = vtk.vtkActor()
        actor1.SetMapper(dataMapper)
        self.ren.RemoveAllViewProps()
        self.vtkWidget.GetRenderWindow().Render()
        self.ren.AddActor(actor)
        self.ren.AddActor(actor1)
        self.ren.ResetCamera()
        self.show()
        self.iren.Initialize()



    def pred(self):
        pcd=self.data_set.predict(self.index)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        vtk_colors = vtk.vtkUnsignedCharArray()
        vtk_colors.SetNumberOfComponents(3)
        vtk_colors.SetName("Colors")
        for color in colors:
            vtk_colors.InsertNextTuple3(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))

        # 创建VTK的vtkPoints对象
        vtk_points = vtk.vtkPoints()
        # 将NumPy数组中的点添加到vtkPoints对象中
        for point in points:
            vtk_points.InsertNextPoint(point[0], point[1], point[2])

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vtk_points)
        if colors.size > 0:
            polydata.GetPointData().SetScalars(vtk_colors)
        glyphFilter = vtk.vtkVertexGlyphFilter()
        glyphFilter.SetInputData(polydata)
        glyphFilter.Update()
        dataMapper = vtk.vtkPolyDataMapper()
        dataMapper.SetInputConnection(glyphFilter.GetOutputPort())
        # Create an actor
        actor = vtk.vtkActor()
        actor.SetMapper(dataMapper)
        actor1 = vtk.vtkActor()
        actor1.SetMapper(dataMapper)
        self.ren.RemoveAllViewProps()
        self.vtkWidget.GetRenderWindow().Render()
        self.ren.AddActor(actor)
        self.ren.AddActor(actor1)
        self.ren.ResetCamera()
        self.show()
        self.iren.Initialize()
        self.data_set.display_pc(pcd)
    def next(self):
        self.index+=1
        self.display(self.index)
    def last(self):
        self.index -= 1
        if self.index<=1:
            self.index=0
        self.display(self.index)
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Mywindow()
    window.show()
    sys.exit(app.exec_())
