from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGroupBox, QVBoxLayout, QMdiSubWindow, QTextEdit, QLayout


class Module(QMdiSubWindow):
    def __init__(self, main_window, name):
        super().__init__()
        self.title = name
        self.setWindowTitle(name)

        self.module = QGroupBox()
        self.layout = QVBoxLayout()
        self.module.setLayout(self.layout)
        self.setWidget(self.module)

        main_window.add_tile(self)
        self.show()
        main_window.tile()

