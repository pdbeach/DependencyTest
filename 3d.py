import sys
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

        # ---------------------------
        # 1) Create random float data
        # ---------------------------
        # shape = (1024, 136, 144)
        # We'll keep it smaller for demonstration, but you can revert to the full shape if desired.
        self.data = np.random.rand(1024, 136, 144).astype(np.float32)
        self.nx, self.ny, self.nz = self.data.shape

        # 2) Determine dynamic vmin/vmax from the data
        self.vmin = float(self.data.min())   # typically ~0.0
        self.vmax = float(self.data.max())   # typically ~1.0

        # 3) Rotation / Transform
        self.xRot = 0.0
        self.yRot = 0.0
        self.lastMouseX = 0
        self.lastMouseY = 0

        # 4) Clipping plane
        self.cutPlaneY = 0.0

        # 5) Subsampling factor (to keep rendering fast)
        #    Drawing every voxel is ~1024*136*144 ~ 20 million points.
        #    We'll skip every 8th voxel in each dimension -> ~5,000 times fewer points.
        self.step = 8

    def initializeGL(self):
        """
        Called once before the first call to paintGL().
        Setup any OpenGL state here.
        """
        glClearColor(0.2, 0.2, 0.2, 1.0)
        glEnable(GL_DEPTH_TEST)
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

        # Move camera back so we see the volume
        # If it's too close, you might only see a corner.
        glTranslated(0.0, 0.0, -12.0)

        # Apply rotations from mouse
        glRotated(self.xRot, 1, 0, 0)
        glRotated(self.yRot, 0, 1, 0)

        # Define the clipping plane along y = cutPlaneY
        # plane eqn: A*x + B*y + C*z + D = 0
        # for y = k -> (0,1,0, -k)
        plane_eq = [0.0, 1.0, 0.0, -self.cutPlaneY]
        glClipPlane(GL_CLIP_PLANE0, plane_eq)

        # Draw the subsampled data
        self.drawSubsampledData()

    def drawSubsampledData(self):
        """
        Draw the 3D volume as a point cloud,
        skipping every 'step'-th voxel to avoid freezing.
        """
        nx, ny, nz = self.nx, self.ny, self.nz
        step = self.step

        # Center volume around origin before scaling
        glTranslatef(-nx/2.0, -ny/2.0, -nz/2.0)
        
        # Scale volume to ~[-1..1]^3 range
        glScalef(2.0/nx, 2.0/ny, 2.0/nz)

        glBegin(GL_POINTS)
        for x in range(0, nx, step):
            for y in range(0, ny, step):
                for z in range(0, nz, step):
                    val = self.data[x, y, z]
                    # clamp to [vmin, vmax]
                    if val < self.vmin:
                        val = self.vmin
                    elif val > self.vmax:
                        val = self.vmax
                    # map to [0..1] for gray color
                    gray = (val - self.vmin) / (self.vmax - self.vmin)
                    glColor3f(gray, gray, gray)
                    glVertex3f(x, y, z)
        glEnd()

    def mousePressEvent(self, event):
        """
        Track the last mouse position for rotation.
        """
        self.lastMouseX = event.x()
        self.lastMouseY = event.y()

    def mouseMoveEvent(self, event):
        """
        When mouse is moved while pressed, update rotation angles.
        """
        dx = event.x() - self.lastMouseX
        dy = event.y() - self.lastMouseY

        self.xRot += dy * 0.5
        self.yRot += dx * 0.5

        self.lastMouseX = event.x()
        self.lastMouseY = event.y()
        self.update()

    def setCutPlaneY(self, val):
        """
        Slider function to set cutPlaneY.
        Maps slider -20..20 => -2..2
        """
        self.cutPlaneY = val / 10.0
        self.update()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Large 3D Float32 Data - Grayscale Viewer")

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Our custom GL widget
        self.glWidget = MyGLWidget(self)

        # Vertical slider for Y clipping
        self.slider = QSlider(Qt.Vertical)
        self.slider.setRange(-20, 20)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.glWidget.setCutPlaneY)

        # Label
        self.label = QLabel("Cut Y")
        self.label.setAlignment(Qt.AlignCenter)

        # Layout
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