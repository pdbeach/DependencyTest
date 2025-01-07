import sys
import numpy as np

from PyQt5.QtWidgets import QApplication, QMainWindow
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
    and reshapes it to the given shape.
    """
    # Load raw float32 data
    volume_data = np.fromfile(file_path, dtype=dtype)
    if volume_data.size != np.prod(shape):
        raise ValueError(f"Volume data size {volume_data.size} "
                         f"does not match expected shape {shape}.")
    volume_data = volume_data.reshape(shape)

    # Optional: Normalize data to [0,1] for easier visualization
    # volume_data = (volume_data - volume_data.min()) / (volume_data.ptp() + 1e-8)

    return volume_data


##############################################################################
# 2. Create a custom QOpenGLWidget for volume rendering
##############################################################################
class VolumeRenderWidget(QOpenGLWidget):
    def __init__(self, volume_data, parent=None):
        super().__init__(parent)
        self.volume_data = volume_data
        self.texture_id = None
        self.shader_program = None
        self.vao = None
        self.vbo = None
        self.screen_quad_coords = np.array([
            -1.0, -1.0, 0.0,
             1.0, -1.0, 0.0,
            -1.0,  1.0, 0.0,
             1.0,  1.0, 0.0
        ], dtype=np.float32)

        # For rotating the volume in the fragment shader, you might
        # want to pass transformation matrices, etc.
        self.angle_x = 0.0
        self.angle_y = 0.0

        # Decide how many steps you want to take in the ray-marching
        self.num_steps = 256

    def initializeGL(self):
        """Set up shaders, create the 3D texture, and configure OpenGL state."""
        # 1) Compile and link shaders
        self.shader_program = self.create_shader_program()

        # 2) Create VAO & VBO for a fullscreen quad
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.screen_quad_coords.nbytes,
                     self.screen_quad_coords, GL_STATIC_DRAW)

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

        # The volume data shape is (D, H, W) = (1024, 136, 144) in this example
        d, h, w = self.volume_data.shape
        # Upload data to the GPU. Data is float32, so internal format can be GL_R32F
        glTexImage3D(
            GL_TEXTURE_3D, 0, GL_R32F,
            w, h, d, 0,
            GL_RED, GL_FLOAT, self.volume_data
        )

        # Unbind everything
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
        glBindTexture(GL_TEXTURE_3D, 0)

        # General OpenGL settings
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glEnable(GL_DEPTH_TEST)

    def paintGL(self):
        """
        Render the volume by performing ray-marching in the fragment shader.
        We'll draw a fullscreen quad, and let the fragment shader compute colors.
        """
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(self.shader_program)

        # Pass uniforms
        glUniform1i(glGetUniformLocation(self.shader_program, "volumeTex"), 0)
        glUniform1f(glGetUniformLocation(self.shader_program, "numSteps"), float(self.num_steps))
        glUniform1f(glGetUniformLocation(self.shader_program, "angleX"), self.angle_x)
        glUniform1f(glGetUniformLocation(self.shader_program, "angleY"), self.angle_y)

        # Bind the 3D texture to texture unit 0
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_3D, self.texture_id)

        # Draw the fullscreen quad (2 triangles)
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

        # Unbind
        glBindVertexArray(0)
        glUseProgram(0)

    def resizeGL(self, w, h):
        """Update the viewport."""
        glViewport(0, 0, w, h)

    def create_shader_program(self):
        """
        Create a simple raycasting shader program.
        """
        vertex_src = r"""
        #version 330 core
        layout(location = 0) in vec3 a_position;

        void main()
        {
            gl_Position = vec4(a_position, 1.0);
        }
        """

        fragment_src = r"""
        #version 330 core

        // The 3D texture (our volume)
        uniform sampler3D volumeTex;

        // Number of steps to take in raymarch
        uniform float numSteps;

        // Simple angles to rotate the ray in volume space
        uniform float angleX;
        uniform float angleY;

        out vec4 fragColor;

        // A simple rotation around X
        mat3 rotX(float a) {
            float c = cos(a);
            float s = sin(a);
            return mat3(
                1.0, 0.0, 0.0,
                0.0,   c,  -s,
                0.0,   s,   c
            );
        }

        // A simple rotation around Y
        mat3 rotY(float a) {
            float c = cos(a);
            float s = sin(a);
            return mat3(
                c, 0.0, s,
                0.0, 1.0, 0.0,
               -s, 0.0, c
            );
        }

        void main()
        {
            // Ray starts at the "front" of the volume. Let's say -1.0 in Z
            // and travels to +1.0 in Z. We'll march from z=-1 to z=+1 in clip space.
            // We'll then transform these positions into [0..1]^3 texture space.

            // For each fragment, define a ray from zNear to zFar in model space:
            // We consider x,y in [-1,1], z from -1..1.
            vec2 uv = (gl_FragCoord.xy / vec2(textureSize(volumeTex, 0).xy));
            // But gl_FragCoord is in screen space, so let's do a simpler approach:
            // We'll treat gl_FragCoord.x and y as if we had a full-screen quad from [-1..1].

            // For a real example, you would typically pass the NDC or do a more precise approach.
            // We'll just approximate a direct mapping from screen coords to [-1..1].
            // This is a minimal example, so let's do it in a contrived manner:
            vec2 screenSize = vec2(800, 600); // <--- might be replaced by uniform
            vec2 xyNDC = (gl_FragCoord.xy / screenSize) * 2.0 - 1.0;

            // We'll define the entry point in model space (z = -1) and exit point (z = +1).
            // Then rotate them around X and Y to see the volume from different angles.
            mat3 rotation = rotY(angleY) * rotX(angleX);

            // Start and end in [-1,1]
            vec3 startPos = rotation * vec3(xyNDC, -1.0);
            vec3 endPos   = rotation * vec3(xyNDC,  1.0);

            // Step direction
            vec3 rayDir = (endPos - startPos) / numSteps;

            // Accumulate color
            vec3 color = vec3(0.0);
            float alpha = 0.0;

            // March through the volume
            vec3 currentPos = startPos;
            for (int i = 0; i < int(numSteps); i++)
            {
                // Transform currentPos from [-1..1] to [0..1]
                vec3 texCoord = currentPos * 0.5 + 0.5;

                // If texCoord is outside [0..1], skip
                if (any(lessThan(texCoord, vec3(0.0))) || any(greaterThan(texCoord, vec3(1.0)))) {
                    currentPos += rayDir;
                    continue;
                }

                // Sample the volume
                float val = texture(volumeTex, texCoord).r;
                // Simple grayscale color
                vec3 sampleColor = vec3(val);

                // Simple front-to-back compositing
                float sampleAlpha = val * 0.05; // scale alpha
                color = mix(color, sampleColor, sampleAlpha);
                alpha = alpha + sampleAlpha * (1.0 - alpha);

                // Early ray termination if alpha is close to 1
                if (alpha >= 0.95) {
                    break;
                }

                currentPos += rayDir;
            }

            fragColor = vec4(color, alpha);
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
        self.setWindowTitle("3D Volume Renderer (PyQt5 + OpenGL)")

        # Create the volume render widget
        self.render_widget = VolumeRenderWidget(volume_data, self)
        self.setCentralWidget(self.render_widget)

        # Optional: you can add sliders or mouse events to rotate the volume
        self.resize(800, 600)

    # Example of a key press event to adjust angles
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left:
            self.render_widget.angle_y -= 0.1
            self.render_widget.update()
        elif event.key() == Qt.Key_Right:
            self.render_widget.angle_y += 0.1
            self.render_widget.update()
        elif event.key() == Qt.Key_Up:
            self.render_widget.angle_x -= 0.1
            self.render_widget.update()
        elif event.key() == Qt.Key_Down:
            self.render_widget.angle_x += 0.1
            self.render_widget.update()
        else:
            super().keyPressEvent(event)


def main():
    # Tell Qt we want an OpenGL 3.3 Core context (for modern GLSL)
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)

    # Load your volume (float32) from a file
    volume_file = "volume_data.bin"  # Adjust to your path
    volume_shape = (1024, 136, 144)  # Adjust if needed
    volume_data = load_volume(volume_file, shape=volume_shape, dtype=np.float32)

    window = MainWindow(volume_data)
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()