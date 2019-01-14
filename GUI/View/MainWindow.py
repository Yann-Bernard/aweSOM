from PyQt5.QtWidgets import QMainWindow


class MainWindow(QMainWindow):
    def __init__(self, application_name):
        super().__init__()
        self.title = application_name
        self.initialize_ui()

    def initialize_ui(self):
        self.setWindowTitle(self.title)
        self.showMaximized()
