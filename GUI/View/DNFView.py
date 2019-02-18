from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QLabel

from GUI.View.Module import Module
from Parameters import input_path, image_name


class DNFView(Module):
    def __init__(self, main_window):
        super().__init__(main_window, "DNFView")
        self.img = QLabel(self)
        pixmap = QPixmap(input_path+image_name)
        pixmap = pixmap.scaled(300, 300)
        self.img.setPixmap(pixmap)
        self.layout.addWidget(self.img)

    def set_image(self, imgQt):
        pixmap = QPixmap.fromImage(imgQt)
        pixmap = pixmap.scaled(300, 300)
        self.img.setPixmap(pixmap)


