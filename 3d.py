import sys
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QHBoxLayout,
    QVBoxLayout, QWidget, QSlider, QLabel
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
    and reshapes it to the given shape: (Z, Y, X).
    """
    # Load raw float32 data
    volume_data = np.fromfile(file_path, dtype=dtype)
    if volume_data.size != np.prod(shape):
        raise ValueError(
            f"Volume data size {volume_data.size} "
            f"does not match expected shape {shape}."
        )
    volume_data = volume_data.reshape(shape)

    # Optional: normalize data to [0,1] for easier visualization
    # volume_data = (volume_data - volume_data.min()) / (volume_data.ptp() + 1e-8)

    return volume_data


##############################################################################
# 2. A QOpenGLWidget subclass that displays a single X-slice
##############################################################################
class VolumeSliceWidget(QOpenGLWidget):
    """
    Displays a cross-section of the volume in the y-z plane
    at a fixed x index.
    """
    def __init__(self, volume_data, parent=None):
        super().__init__(parent)
        self.volume_data = volume_data
        self.texture_id = None
        self.shader_program = None
        self.vao = None
        self.vbo = None

        # volume_data.shape = (z, y, x) = (1024, 136, 144)
        self.depth, self.height, self.width = volume_data.shape

        # Fullscreen quad positions (x, y in NDC, plus z=0)
        self.screen_quad_coords = np.array([
            -1.0, -1.0, 0.0,   # bottom-left
             1.0, -1.0, 0.0,   # bottom-right
            -1.0,  1.0, 0.0,   # top-left
             1.0,  1.0, 0.0    # top-right
        ], dtype=np.float32)

        # Which x-slice do we show? Range is [0, width-1]
        self.x_slice_index = self.width // 2  # start in the middle

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

        # Position attribute (location = 0)
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

        # Upload data to the GPU (internal format = GL_R32F)
        glTexImage3D(
            GL_TEXTURE_3D,
            0,
            GL_R32F,
            self.width,      # X
            self.height,     # Y
            self.depth,      # Z
            0,
            GL_RED,
            GL_FLOAT,
            self.volume_data
        )

        # Unbind
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
        glBindTexture(GL_TEXTURE_3D, 0)

        # General OpenGL settings
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glDisable(GL_DEPTH_TEST)

    def paintGL(self):
        """
        Render the selected X-slice by sampling the 3D texture at:
        (z, y, x = x_slice_index).
        """
        glClear(GL_COLOR_BUFFER_BIT)

        glUseProgram(self.shader_program)
        glBindVertexArray(self.vao)

        # Bind the 3D texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_3D, self.texture_id)
        glUniform1i(glGetUniformLocation(self.shader_program, "volumeTex"), 0)

        # Pass the x-slice uniform
        x_loc = glGetUniformLocation(self.shader_program, "xIndex")
        glUniform1f(x_loc, float(self.x_slice_index))

        # Pass width size for normalization
        x_size_loc = glGetUniformLocation(self.shader_program, "xSize")
        glUniform1f(x_size_loc, float(self.width))

        # Draw the fullscreen quad
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

        glBindVertexArray(0)
        glUseProgram(0)

    def resizeGL(self, w, h):
        """Update the viewport."""
        glViewport(0, 0, w, h)

    def create_shader_program(self):
        """
        Create a simple shader that displays a slice in the x dimension.
        We'll interpret the slice at xIndex, and map the
        screen quad to y,z axes.
        
        volumeTex is sampled as (z, y, x).
        """
        vertex_src = r"""
        #version 330 core
        layout(location = 0) in vec3 a_position;

        // We'll pass UV coordinates to the fragment shader
        out vec2 vUV;

        void main()
        {
            // a_position in [-1..1]
            // Map to [0..1] for vUV
            // We'll interpret vUV.x => y, vUV.y => z
            vUV = (a_position.xy * 0.5) + 0.5;
            gl_Position = vec4(a_position, 1.0);
        }
        """

        fragment_src = r"""
        #version 330 core

        in vec2 vUV;
        out vec4 fragColor;

        // The 3D volume (z, y, x)
        uniform sampler3D volumeTex;

        // The x dimension slice
        uniform float xIndex;
        // The total width
        uniform float xSize;

        void main()
        {
            // xIndex in [0..xSize-1], convert to [0..1]
            float xTex = xIndex / (xSize - 1.0);

            // vUV.x => y in [0..1]
            // vUV.y => z in [0..1]

            // sample (z, y, x) = (vUV.y, vUV.x, xTex)
            float val = texture(volumeTex, vec3(vUV.y, vUV.x, xTex)).r;

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
        self.setWindowTitle("Slice Through X-Axis (PyQt5 + OpenGL)")

        # Main container
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Vertical layout: top = GL widget, bottom = slider
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # 1) The volume slice widget
        self.render_widget = VolumeSliceWidget(volume_data, self)
        main_layout.addWidget(self.render_widget, stretch=1)

        # 2) A small horizontal layout for the slider + label
        slider_layout = QHBoxLayout()

        # Add a label for the slider
        self.label = QLabel("X-slice:")
        slider_layout.addWidget(self.label)

        # The slider controlling x slices
        self.slider = QSlider(Qt.Horizontal)
        # x dimension is volume_data.shape[2]
        _, _, w = volume_data.shape
        self.slider.setRange(0, w - 1)
        self.slider.setValue(w // 2)  # start in the middle
        # Make it a bit narrower
        self.slider.setFixedWidth(300)

        # Connect slider to handler
        self.slider.valueChanged.connect(self.on_xslice_changed)

        slider_layout.addWidget(self.slider)

        # Possibly add a numeric label showing the current slice
        self.current_slice_label = QLabel(f"{self.slider.value()}")
        slider_layout.addWidget(self.current_slice_label)

        main_layout.addLayout(slider_layout)

        self.resize(800, 600)

    def on_xslice_changed(self, value):
        """Update the x slice index in the OpenGL widget and refresh."""
        self.render_widget.x_slice_index = value
        self.current_slice_label.setText(str(value))
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
    # (z, y, x) = (1024, 136, 144)
    volume_shape = (1024, 136, 144)
    volume_data = load_volume(volume_file, shape=volume_shape, dtype=np.float32)

    window = MainWindow(volume_data)
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()