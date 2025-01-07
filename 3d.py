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
    glPointSize,
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

        # 1) Create random float data in [0,1]
        # shape = (1024, 136, 144)
        # That’s about 20 million voxels, so we must subsample.
        self.data = np.random.rand(1024, 136, 144).astype(np.float32)
        self.nx, self.ny, self.nz = self.data.shape

        # 2) Determine dynamic vmin/vmax from the data
        self.vmin = float(self.data.min())   # ~0.0
        self.vmax = float(self.data.max())   # ~1.0

        # 3) Rotation / Transform
        self.xRot = 0.0
        self.yRot = 0.0
        self.lastMouseX = 0
        self.lastMouseY = 0

        # 4) Clipping plane along y
        self.cutPlaneY = 0.0

        # 5) Subsampling factor
        #    We'll draw 1 voxel out of every 8 in each dimension.
        #    This cuts from ~20 million to ~39k points.
        self.step = 8

    def initializeGL(self):
        """
        Called once before the first call to paintGL().
        Setup any OpenGL state here.
        """
        glClearColor(0.2, 0.2, 0.2, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CLIP_PLANE0)

        # Make points a bit larger so they're visible
        glPointSize(3.0)

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

        # Move the camera back some (if too close, you may not see much)
        glTranslated(0.0, 0.0, -12.0)

        # Apply rotations from mouse
        glRotated(self.xRot, 1, 0, 0)
        glRotated(self.yRot, 0, 1, 0)

        # Define the clipping plane along y = cutPlaneY
        # plane eqn: A*x + B*y + C*z + D = 0
        # for y = k => (0,1,0,-k)
        plane_eq = [0.0, 1.0, 0.0, -self.cutPlaneY]
        glClipPlane(GL_CLIP_PLANE0, plane_eq)

        # Draw the subsampled data
        self.drawSubsampledData()

    def drawSubsampledData(self):
        """
        Draw the 3D volume as points, skipping every 'step' voxel,
        using a color map that is obviously not uniform.
        """
        nx, ny, nz = self.nx, self.ny, self.nz
        step = self.step

        # Move the volume so it's roughly centered around origin
        glTranslatef(-nx/2.0, -ny/2.0, -nz/2.0)

        # Scale it to roughly fill [-1..1]^3
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

                    # Map [vmin..vmax] to [0..1]
                    c = (val - self.vmin) / (self.vmax - self.vmin)

                    # Instead of grayscale, let's make a simple color gradient:
                    # e.g., c in Red/Green, plus constant Blue.
                    # That way, random data is obviously multi-colored.
                    r = c
                    g = 1.0 - c
                    b = 0.5
                    glColor3f(r, g, b)

                    glVertex3f(x, y, z)
        glEnd()

    def mousePressEvent(self, event):
        """
        Remember the last mouse position for rotation.
        """
        self.lastMouseX = event.x()
        self.lastMouseY = event.y()

    def mouseMoveEvent(self, event):
        """
        When the mouse moves while pressed, update rotation angles.
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
        Slider function for Y clipping plane.
        Maps slider range [-20..20] => [-2..2].
        """
        self.cutPlaneY = val / 10.0
        self.update()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Large 3D Float32 Data - Colorful Viewer")

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Our custom OpenGL widget
        self.glWidget = MyGLWidget(self)

        # Vertical slider to control y-clipping
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