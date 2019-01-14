from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QLabel

from GUI.View.Module import Module
from Parameters import input_path, image_name


class Image(Module):
    def __init__(self, main_window):
        super().__init__(main_window, "Input Image")
        self.img = QLabel(self)
        pixmap = QPixmap(input_path+image_name)
        self.img.setPixmap(pixmap)
        self.layout.addWidget(self.img)
        self.showMaximized()
