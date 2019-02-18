from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QLabel
from PyQt5.uic.properties import QtCore

from GUI.View.Module import Module
from Parameters import input_path, image_name


class Image(Module):
    def __init__(self, main_window):
        super().__init__(main_window, "Input Image")
        self.img = QLabel(self)
        pixmap = QPixmap(input_path+image_name)
        pixmap = pixmap.scaled(335, 365)
        self.img.setPixmap(pixmap)
        self.layout.addWidget(self.img)

    def set_image(self, imgQt):
        pixmap = QPixmap.fromImage(imgQt)
        pixmap = pixmap.scaled(335, 365)
        self.img.setPixmap(pixmap)
