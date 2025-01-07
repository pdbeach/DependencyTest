import sys
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QHBoxLayout,
    QSlider, QWidget, QVBoxLayout, QLabel
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QSurfaceFormat
from PyQt5.QtOpenGL import QOpenGLWidget

# PyOpenGL imports
from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram

##############################################################################
# 1. Load the volume data (float32) from file
##############################################################################
def load_volume(file_path, shape=(1024, 136, 144), dtype=np.float32):
    """
    Reads a binary file containing a 3D volume of float32 values
    and reshapes it to the given shape: (D, H, W).
    """
    # Load raw float32 data
    volume_data = np.fromfile(file_path, dtype=dtype)
    if volume_data.size != np.prod(shape):
        raise ValueError(
            f"Volume data size {volume_data.size} "
            f"does not match expected shape {shape}."
        )
    volume_data = volume_data.reshape(shape)

    # Optional: Normalize data to [0,1] for easier visualization
    # volume_data = (volume_data - volume_data.min()) / (volume_data.ptp() + 1e-8)

    return volume_data


##############################################################################
# 2. A QOpenGLWidget subclass that displays a single Y-slice
##############################################################################
class VolumeSliceWidget(QOpenGLWidget):
    def __init__(self, volume_data, parent=None):
        super().__init__(parent)
        self.volume_data = volume_data
        self.texture_id = None
        self.shader_program = None
        self.vao = None
        self.vbo = None

        # The 3D volume shape is (Depth, Height, Width)
        self.depth, self.height, self.width = volume_data.shape

        # Fullscreen quad positions (x, y in NDC, plus dummy z=0)
        self.screen_quad_coords = np.array([
            -1.0, -1.0, 0.0,   # bottom-left
             1.0, -1.0, 0.0,   # bottom-right
            -1.0,  1.0, 0.0,   # top-left
             1.0,  1.0, 0.0    # top-right
        ], dtype=np.float32)

        # Which slice in the Y dimension do we show?
        # Range is [0, height-1]
        self.slice_index = self.height // 2  # start in the middle

    def initializeGL(self):
        """Set up shaders, create the 3D texture, and configure OpenGL state."""
        # 1) Compile and link shaders
        self.shader_program = self.create_shader_program()

        # 2) Create VAO & VBO for a fullscreen quad
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER,
                     self.screen_quad_coords.nbytes,
                     self.screen_quad_coords,
                     GL_STATIC_DRAW)

        # In this example, we have a single attribute: position (location = 0)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)

        # 3) Create and fill a 3D texture with the volume data
        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_3D, self.texture_id)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)

        # volume_data.shape = (D, H, W) = (1024, 136, 144)
        d, h, w = self.volume_data.shape

        # Upload data to the GPU. Data is float32, so internal format can be GL_R32F
        glTexImage3D(
            GL_TEXTURE_3D,
            0,
            GL_R32F,
            w, h, d,
            0,
            GL_RED,
            GL_FLOAT,
            self.volume_data
        )

        # Unbind to clean up
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
        glBindTexture(GL_TEXTURE_3D, 0)

        # General OpenGL settings
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glDisable(GL_DEPTH_TEST)  # For a simple 2D slice, depth test not needed

    def paintGL(self):
        """
        Render the selected slice by sampling the 3D texture at:
        (x, y=sliceIndex, z) for each screen pixel (mapped to x,z).
        """
        glClear(GL_COLOR_BUFFER_BIT)

        glUseProgram(self.shader_program)
        glBindVertexArray(self.vao)

        # Bind the 3D texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_3D, self.texture_id)
        glUniform1i(glGetUniformLocation(self.shader_program, "volumeTex"), 0)

        # Pass uniforms
        slice_loc = glGetUniformLocation(self.shader_program, "sliceIndex")
        glUniform1f(slice_loc, float(self.slice_index))

        height_loc = glGetUniformLocation(self.shader_program, "heightSize")
        glUniform1f(height_loc, float(self.height))

        # Draw the fullscreen quad
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

        glBindVertexArray(0)
        glUseProgram(0)

    def resizeGL(self, w, h):
        """Update the viewport."""
        glViewport(0, 0, w, h)

    def create_shader_program(self):
        """
        Create a simple shader program that samples from 3D volume
        at a fixed 'y' slice.
        """
        vertex_src = r"""
        #version 330 core
        layout(location = 0) in vec3 a_position;

        // We'll pass UV coordinates to the fragment shader
        out vec2 vUV;

        void main()
        {
            // a_position in [-1..1]
            // Map to [0..1] for x, z
            vUV = (a_position.xy * 0.5) + 0.5;
            gl_Position = vec4(a_position, 1.0);
        }
        """

        fragment_src = r"""
        #version 330 core

        in vec2 vUV;
        out vec4 fragColor;

        // The 3D volume (D,H,W)
        uniform sampler3D volumeTex;

        // Which slice we show in the Y dimension
        uniform float sliceIndex;
        // The total size of the Y dimension
        uniform float heightSize;

        void main()
        {
            // sliceIndex is in [0..heightSize-1]
            // Convert to [0..1]
            float yTex = sliceIndex / (heightSize - 1.0);

            // vUV is in [0..1] for the x and z axes
            // So we sample volumeTex at (x, y, z) = (vUV.x, yTex, vUV.y)
            float val = texture(volumeTex, vec3(vUV.x, yTex, vUV.y)).r;

            // Simple grayscale
            fragColor = vec4(val, val, val, 1.0);
        }
        """

        # Compile shaders
        vert_shader = compileShader(vertex_src, GL_VERTEX_SHADER)
        frag_shader = compileShader(fragment_src, GL_FRAGMENT_SHADER)
        prog = compileProgram(vert_shader, frag_shader)
        return prog


##############################################################################
# 3. Main Window / Application
##############################################################################
class MainWindow(QMainWindow):
    def __init__(self, volume_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Slice Through Y-Axis (PyQt5 + OpenGL)")

        # Central widget with layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # Create the volume slice widget
        self.render_widget = VolumeSliceWidget(volume_data, self)
        layout.addWidget(self.render_widget)

        # Add a horizontal slider to move through the Y dimension
        self.slider = QSlider(Qt.Horizontal)
        # Y dimension is volume_data.shape[1]
        _, h, _ = volume_data.shape
        self.slider.setRange(0, h - 1)
        self.slider.setValue(h // 2)  # start in the middle
        self.slider.valueChanged.connect(self.on_slice_changed)
        layout.addWidget(self.slider)

        # Optionally show the current slice number
        self.label = QLabel(f"Y-slice: {self.slider.value()}")
        layout.addWidget(self.label)

        self.resize(800, 600)

    def on_slice_changed(self, value):
        """Update the slice index in the OpenGL widget and refresh."""
        self.render_widget.slice_index = value
        self.label.setText(f"Y-slice: {value}")
        self.render_widget.update()


def main():
    # Request an OpenGL 3.3 Core profile
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)

    # Load your volume (float32) from a file
    volume_file = "volume_data.bin"  # Adjust to your path
    volume_shape = (1024, 136, 144)  # (Depth, Height, Width)
    volume_data = load_volume(volume_file, shape=volume_shape, dtype=np.float32)

    window = MainWindow(volume_data)
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()