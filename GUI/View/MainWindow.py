from PyQt5.QtWidgets import QMainWindow, QMdiArea


class MainWindow(QMainWindow):
    def __init__(self, application_name):
        super(MainWindow, self).__init__()
        self.title = application_name
        self.mdi = QMdiArea()
        self.initialize_ui()

    def initialize_ui(self):
        self.setCentralWidget(self.mdi)
        self.setWindowTitle(self.title)
        self.showMaximized()

    def add_tile(self, module):
        self.mdi.addSubWindow(module)

    def tile(self):
        self.mdi.tileSubWindows()

