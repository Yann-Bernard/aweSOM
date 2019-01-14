from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGroupBox, QVBoxLayout


class Module(QGroupBox):
    def __init__(self, main_window, name):
        super().__init__(main_window)
        self.title = name
        self.setTitle(name)
        self.setAlignment(Qt.AlignHCenter)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
