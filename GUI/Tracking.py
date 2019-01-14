import sys
from PyQt5.QtWidgets import QApplication

from GUI.View.Image import Image
from GUI.View.MainWindow import MainWindow
from GUI.View.Module import Module

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow("Tracking")
    image = Image(window)
    image2 = Image(window)
    sys.exit(app.exec_())

