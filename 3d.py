limport sys
import numpy as np

# Force Matplotlib to use the Qt5Agg backend (optional, may be automatic)
import matplotlib
matplotlib.use('Qt5Agg')

# Matplotlib canvas for PyQt5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

# PyQt5 imports
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QHBoxLayout,
    QVBoxLayout, QWidget, QSlider, QLabel
)
from PyQt5.QtCore import Qt


##############################################################################
# 1. Load the volume data (float32) from file
##############################################################################
def load_volume(file_path, shape=(1024, 136, 144), dtype=np.float32):
    """
    Reads a binary file containing a 3D volume of float32 values
    and reshapes it to the given shape: (Z, Y, X).

    Make sure 'volume_data.bin' has exactly shape[0]*shape[1]*shape[2] float32s.
    """
    volume_data = np.fromfile(file_path, dtype=dtype)
    if volume_data.size != np.prod(shape):
        raise ValueError(
            f"Volume data size {volume_data.size} "
            f"does not match expected shape {shape}."
        )
    volume_data = volume_data.reshape(shape)

    # Optional: normalize data to [0,1] if desired
    # volume_data = (volume_data - volume_data.min()) / (volume_data.ptp() + 1e-8)

    return volume_data


##############################################################################
# 2. Matplotlib Canvas that displays a single X-slice
##############################################################################
class VolumeSliceCanvas(FigureCanvasQTAgg):
    """
    Displays a cross-section of the volume in the y-z plane
    at a fixed x index using Matplotlib.
    """
    def __init__(self, volume_data, parent=None):
        # Create a fresh Matplotlib Figure
        fig = Figure()
        super().__init__(fig)
        self.setParent(parent)

        self.volume_data = volume_data

        # Rename dimensions to avoid Qt's built-in height() / width() conflicts
        self.z_dim, self.y_dim, self.x_dim = volume_data.shape  # (Z, Y, X)

        # Prepare a Matplotlib Axes
        self.ax = self.figure.subplots()

        # Default x-slice index (start in the middle)
        self.x_slice_index = self.x_dim // 2

        # Display the initial slice
        self.show_slice(self.x_slice_index)

    def show_slice(self, x_index):
        """
        Displays the slice at x_index in the y-z plane, i.e. volume_data[:, :, x_index].
        """
        self.ax.clear()
        slice_data = self.volume_data[:, :, x_index]

        # Show the slice as an image
        self.ax.imshow(slice_data, cmap='gray', origin='lower', aspect='auto')
        self.ax.set_title(f"X-slice = {x_index}")

        # Force an immediate redraw
        self.draw()

    def update_slice(self, x_index):
        """ Update the displayed slice index and redraw. """
        self.x_slice_index = x_index
        self.show_slice(x_index)


##############################################################################
# 3. Main Window / Application
##############################################################################
class MainWindow(QMainWindow):
    def __init__(self, volume_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Slice Through X-Axis (PyQt5 + Matplotlib)")

        # Main container widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Vertical layout: top = Matplotlib canvas, bottom = slider
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # 1) The volume slice canvas (Matplotlib)
        self.slice_canvas = VolumeSliceCanvas(volume_data, self)
        main_layout.addWidget(self.slice_canvas, stretch=1)

        # 2) A small horizontal layout for the slider + label
        slider_layout = QHBoxLayout()

        # Label for the slider
        self.label = QLabel("X-slice:")
        slider_layout.addWidget(self.label)

        # The slider controlling x slices
        self.slider = QSlider(Qt.Horizontal)
        # x dimension is volume_data.shape[2]
        _, _, x_dim = volume_data.shape
        self.slider.setRange(0, x_dim - 1)
        self.slider.setValue(x_dim // 2)  # start in the middle
        self.slider.setFixedWidth(300)

        # Connect slider to handler
        self.slider.valueChanged.connect(self.on_xslice_changed)
        slider_layout.addWidget(self.slider)

        # Numeric label for current slice
        self.current_slice_label = QLabel(f"{self.slider.value()}")
        slider_layout.addWidget(self.current_slice_label)

        main_layout.addLayout(slider_layout)

        self.resize(800, 600)

    def on_xslice_changed(self, value):
        """Update the x slice index in the Matplotlib widget and refresh."""
        self.current_slice_label.setText(str(value))
        self.slice_canvas.update_slice(value)


def main():
    app = QApplication(sys.argv)

    # Adjust to your file/path. 'volume_data.bin' must match shape (1024, 136, 144).
    volume_file = "volume_data.bin"
    volume_shape = (1024, 136, 144)  # (Z, Y, X)
    volume_data = load_volume(volume_file, shape=volume_shape, dtype=np.float32)

    window = MainWindow(volume_data)
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()