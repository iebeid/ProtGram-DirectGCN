import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QMessageBox, QComboBox, QFrame)
from PyQt6.QtCore import Qt, QThread
from PyQt6.QtGui import QFont

# Matplotlib imports for embedding the plot in PyQt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Import the worker class from our other file
from tsne_worker import TSNEWorker


class MplCanvas(FigureCanvas):
    """A custom widget to embed a Matplotlib figure into a PyQt6 application."""

    def __init__(self, fig: Figure, parent=None):
        self.fig = fig
        super().__init__(self.fig)
        self.setParent(parent)
        self.setMinimumSize(400, 300)  # Set a reasonable minimum size


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("t-SNE Embedding Visualizer")
        self.setGeometry(100, 100, 800, 650)

        # A state variable to track if we are showing the initial placeholder
        self._is_placeholder_shown = True

        self.main_layout = QVBoxLayout()
        self.central_widget = QWidget()
        self.central_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.central_widget)

        self.controls_layout = QVBoxLayout()
        self.main_layout.addLayout(self.controls_layout)

        self.embed_type_label = QLabel("1. Select Embedding Type:")
        self.embed_type_combo = QComboBox()
        self.embed_type_combo.addItems(['generic_multiple_sets', 'per_protein', 'per_residue'])
        self.controls_layout.addWidget(self.embed_type_label)
        self.controls_layout.addWidget(self.embed_type_combo)

        self.drop_zone_label = QLabel("\n\n2. Drop .h5 File Here\n\n")
        font = self.drop_zone_label.font()
        font.setPointSize(20)
        self.drop_zone_label.setFont(font)
        self.drop_zone_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drop_zone_label.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken)
        self.drop_zone_label.setLineWidth(2)

        self.plot_container_layout = QVBoxLayout()
        self.plot_container_layout.addWidget(self.drop_zone_label)
        self.main_layout.addLayout(self.plot_container_layout)

        self.statusBar().showMessage("Ready. Select embedding type and drop an HDF5 file.")
        self.setAcceptDrops(True)
        self.setup_worker_thread()

    def _clear_layout(self, layout):
        """A helper function to remove all widgets from a layout."""
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def setup_worker_thread(self):
        """Initializes the worker and the thread it runs on."""
        self.worker = TSNEWorker()
        self.thread = QThread()
        self.worker.moveToThread(self.thread)

        self.worker.progress.connect(self.update_status)
        self.worker.error.connect(self.report_error)
        self.worker.finished.connect(self.display_plot)

        self.thread.started.connect(self.on_thread_started)
        self.thread.finished.connect(self.on_thread_finished)

    def start_processing(self, h5_path: str):
        """Starts the background thread to process the H5 file."""
        if self.thread.isRunning():
            QMessageBox.warning(self, "Busy", "A process is already running. Please wait.")
            return

        self.h5_path_to_process = h5_path

        # Reset the UI to its initial state before starting
        self._clear_layout(self.plot_container_layout)
        self.plot_container_layout.addWidget(self.drop_zone_label)
        self._is_placeholder_shown = True

        self.thread.start()

    def on_thread_started(self):
        """Called when the thread starts; triggers the worker's main task."""
        embedding_type = self.embed_type_combo.currentText()
        self.update_status(f"Thread started. Processing '{os.path.basename(self.h5_path_to_process)}'...")
        self.worker.start_tsne_processing(self.h5_path_to_process, embedding_type)

    def on_thread_finished(self):
        """Called when the thread finishes."""
        self.update_status("Processing finished. Ready for new file.")

    def dragEnterEvent(self, event):
        mime_data = event.mimeData()
        if mime_data.hasUrls() and len(mime_data.urls()) == 1:
            url = mime_data.urls()[0]
            if url.isLocalFile() and url.toLocalFile().endswith('.h5'):
                event.acceptProposedAction()

    def dropEvent(self, event):
        """Handles the event when an object is dropped onto the window."""
        url = event.mimeData().urls()[0]  # Corrected method name
        filepath = url.toLocalFile()
        self.start_processing(filepath)

    def update_status(self, message: str):
        self.statusBar().showMessage(message)

    def report_error(self, error_message: str):
        QMessageBox.critical(self, "Error", error_message)
        self.thread.quit()

    def display_plot(self, fig: Figure):
        """Receives a figure from the worker and displays it."""
        # If the placeholder "Drop file here" label is showing, clear it first.
        if self._is_placeholder_shown:
            self._clear_layout(self.plot_container_layout)
            self._is_placeholder_shown = False  # Flip the state

        # Create a canvas widget for the new plot and add it to the layout.
        canvas = MplCanvas(fig, self)
        self.plot_container_layout.addWidget(canvas)
        self.update_status("Plot generated successfully.")

        # For non-streaming plots, the job is done, so we can stop the thread.
        if self.embed_type_combo.currentText() != 'per_residue':
            self.thread.quit()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
