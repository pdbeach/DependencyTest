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
    glDisable,
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

        # --- Volume Data ---
        # Replace this with actual loading code for your float32 data.
        # For example: self.data = np.load("my_volume_data.npy")
        self.nx, self.ny, self.nz = 32, 32, 32
        self.data = np.random.rand(self.nx, self.ny, self.nz).astype(np.float32)

        # We will clamp data to [0, 1.05] when drawing.
        self.vmin = 0.0
        self.vmax = 1.05

        # --- Transformation / Rotation ---
        self.xRot = 0.0
        self.yRot = 0.0
        self.lastMouseX = 0
        self.lastMouseY = 0

        # --- Clipping plane along Y ---
        self.cutPlaneY = 0.0  # default slider value

    def initializeGL(self):
        """
        Called once before the first call to paintGL().
        Setup any OpenGL state here.
        """
        glClearColor(0.2, 0.2, 0.2, 1.0)
        glEnable(GL_DEPTH_TEST)

        # Enable the clipping plane
        glEnable(GL_CLIP_PLANE0)

    def resizeGL(self, w, h):
        """
        Called upon window resizing. We set up our projection matrix here.
        """
        if h == 0:
            h = 1
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        # Simple perspective transformation
        # left, right, bottom, top, near, far
        glFrustum(-1, 1, -1, 1, 1.0, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        """
        Called every time the widget is repainted. All drawing code goes here.
        """
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Move the data away from the camera
        glTranslated(0, 0, -6)

        # Apply rotations (mouse-based)
        glRotated(self.xRot, 1, 0, 0)
        glRotated(self.yRot, 0, 1, 0)

        # Define the clipping plane along y = cutPlaneY
        #   Plane eqn: A*x + B*y + C*z + D = 0
        #   For a plane at y = k:  y - k = 0  =>  (0, 1, 0, -k)
        plane_eq = [0.0, 1.0, 0.0, -self.cutPlaneY]
        glClipPlane(GL_CLIP_PLANE0, plane_eq)

        # Draw your 3D data
        self.drawFloat32Data()

    def drawFloat32Data(self):
        """
        Draw all voxels as a point cloud, colored by data value in [vmin, vmax].
        This is a basic example. For large volumes, you’ll need a more advanced approach.
        """
        nx, ny, nz = self.nx, self.ny, self.nz

        # Center and scale the data so it fits roughly in [-1,1]^3
        # (This is arbitrary; adjust as needed)
        glTranslatef(-nx / 2.0, -ny / 2.0, -nz / 2.0)
        glScalef(2.0 / nx, 2.0 / ny, 2.0 / nz)

        glBegin(GL_POINTS)
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    val = self.data[x, y, z]
                    # Clamp value to [vmin, vmax]
                    if val < self.vmin:
                        val = self.vmin
                    elif val > self.vmax:
                        val = self.vmax
                    gray = (val - self.vmin) / (self.vmax - self.vmin)
                    # Use gray to set color (grayscale)
                    glColor3f(gray, gray, gray)
                    glVertex3f(x, y, z)
        glEnd()

    def mousePressEvent(self, event):
        """
        Record the last mouse position on press, to handle rotations.
        """
        self.lastMouseX = event.x()
        self.lastMouseY = event.y()

    def mouseMoveEvent(self, event):
        """
        When the mouse moves while pressed, compute how far it moved and update rotations.
        """
        dx = event.x() - self.lastMouseX
        dy = event.y() - self.lastMouseY

        # Update stored rotation angles (tweak sensitivity as you like)
        self.xRot += dy * 0.5
        self.yRot += dx * 0.5

        self.lastMouseX = event.x()
        self.lastMouseY = event.y()

        # Schedule a redraw
        self.update()

    def setCutPlaneY(self, val):
        """
        Slider calls this to update the cut-plane position.
        Suppose the slider range is [-20..20], we map that to [-2..2] by dividing by 10.0
        Adjust as you see fit.
        """
        self.cutPlaneY = val / 10.0
        self.update()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Float32 Data Viewer with Y-Axis Cut Slider")

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create the custom OpenGL widget
        self.glWidget = MyGLWidget(self)

        # Create a slider to cut along the Y axis
        self.slider = QSlider(Qt.Vertical)
        # For example, -20..20 => -2..2 if dividing by 10
        self.slider.setRange(-20, 20)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.glWidget.setCutPlaneY)

        # Create a label
        self.label = QLabel("Cut Y")
        self.label.setAlignment(Qt.AlignCenter)

        # Layout for the slider
        slider_layout = QVBoxLayout()
        slider_layout.addWidget(self.label)
        slider_layout.addWidget(self.slider)

        # Main layout
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