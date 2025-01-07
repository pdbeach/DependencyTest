import sys
import math

import numpy as np

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
    QSlider,
    QLabel
)
from PyQt5.QtCore import Qt
from PyQt5.QtOpenGL import QOpenGLWidget

from OpenGL.GL import (
    glClearColor,
    glClear,
    glEnable,
    glMatrixMode,
    glLoadIdentity,
    glFrustum,
    glTranslated,
    glRotated,
    glBegin,
    glEnd,
    glVertex3f,
    glColor3f,
    glClipPlane,
    glScalef,
    glTranslatef,
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_CLIP_PLANE0,
    GL_MODELVIEW,
    GL_PROJECTION,
    GL_POINTS
)

class MyGLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(800, 600)

        # Suppose your data is shape (1024, 136, 144)
        # Typically loaded with: self.data = np.load("my_volume_data.npy")
        # For demonstration, let's just create dummy data:
        nx, ny, nz = 1024, 136, 144
        self.data = np.random.rand(nx, ny, nz).astype(np.float32)

        # We'll store shape info
        self.nx, self.ny, self.nz = self.data.shape

        # Display range [0, 1.05]
        self.vmin = 0.0
        self.vmax = 1.05

        # Rotation
        self.xRot = 0.0
        self.yRot = 0.0
        self.lastMouseX = 0
        self.lastMouseY = 0

        # Clipping plane Y
        self.cutPlaneY = 0.0

        # Subsampling factor (skip points to reduce load)
        self.step = 4  # try adjusting this

    def initializeGL(self):
        glClearColor(0.2, 0.2, 0.2, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CLIP_PLANE0)

    def resizeGL(self, w, h):
        if h == 0:
            h = 1
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glFrustum(-1, 1, -1, 1, 1.0, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Move the data away from camera
        glTranslated(0, 0, -6)

        # Apply rotations
        glRotated(self.xRot, 1, 0, 0)
        glRotated(self.yRot, 0, 1, 0)

        # Clipping plane (y = cutPlaneY)
        plane_eq = [0.0, 1.0, 0.0, -self.cutPlaneY]
        glClipPlane(GL_CLIP_PLANE0, plane_eq)

        # Draw data as subsampled point cloud
        self.drawDataSubsampled()

    def drawDataSubsampled(self):
        """
        Render the data as points, skipping every `self.step` voxel.
        """
        nx, ny, nz = self.nx, self.ny, self.nz

        # Center and scale volume to roughly [-1..1]^3
        glTranslatef(-nx / 2.0, -ny / 2.0, -nz / 2.0)
        glScalef(2.0 / nx, 2.0 / ny, 2.0 / nz)

        step = self.step

        glBegin(GL_POINTS)
        for x in range(0, nx, step):
            for y in range(0, ny, step):
                for z in range(0, nz, step):
                    val = self.data[x, y, z]
                    # Clamp
                    if val < self.vmin:
                        val = self.vmin
                    elif val > self.vmax:
                        val = self.vmax
                    gray = (val - self.vmin) / (self.vmax - self.vmin)
                    glColor3f(gray, gray, gray)
                    glVertex3f(x, y, z)
        glEnd()

    def mousePressEvent(self, event):
        self.lastMouseX = event.x()
        self.lastMouseY = event.y()

    def mouseMoveEvent(self, event):
        dx = event.x() - self.lastMouseX
        dy = event.y() - self.lastMouseY
        self.xRot += dy * 0.5
        self.yRot += dx * 0.5
        self.lastMouseX = event.x()
        self.lastMouseY = event.y()
        self.update()

    def setCutPlaneY(self, val):
        # map slider -20..20 => -2..2
        self.cutPlaneY = val / 10.0
        self.update()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Float32 Data Viewer with Y-Axis Cut Slider")

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # GL widget
        self.glWidget = MyGLWidget(self)

        # Slider
        self.slider = QSlider(Qt.Vertical)
        self.slider.setRange(-20, 20)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.glWidget.setCutPlaneY)

        # Label
        self.label = QLabel("Cut Y")
        self.label.setAlignment(Qt.AlignCenter)

        # Layouts
        slider_layout = QVBoxLayout()
        slider_layout.addWidget(self.label)
        slider_layout.addWidget(self.slider)

        main_layout = QHBoxLayout(central_widget)
        main_layout.addLayout(slider_layout)
        main_layout.addWidget(self.glWidget)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()