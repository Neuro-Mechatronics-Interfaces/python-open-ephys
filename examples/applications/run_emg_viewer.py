import sys

from PyQt5.QtWidgets import QApplication

from pyoephys.applications import EMGViewer


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = EMGViewer()
    viewer.show()
    sys.exit(app.exec_())
