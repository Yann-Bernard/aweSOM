import threading

import numpy as np
import sys

from PIL.ImageQt import ImageQt
from PyQt5.QtWidgets import QApplication

from Data.ColoredImage import ColoredImage
from Data.Images import ImageData
from CMD.Single_Run import display_som
from GUI.View.Image import Image
from GUI.View.MainWindow import MainWindow
from GUI.View.MapView import MapView
from GUI.View.Module import Module
from GUI.View.ReconstructedImage import ReconstructedImage
from Models.NPSOM.Connections import kohonen
from Models.SOM import SOM
from Parameters import *


def run():
    img = ColoredImage()
    np.random.seed(1024)
    data = img.data
    epoch_time = len(data)
    nb_iter = epoch_time * epoch_nbr
    som = SOM(data, kohonen())

    for i in range(nb_iter):
        # The training vector is chosen randomly
        if i % epoch_time == 0:
            som.generate_random_list()
            tile2.set_image(ImageQt(img.compress(som)))
            tile3.set_image(ImageQt(img.display_som(som.get_som_as_list())))
            print("Epoch : ", (i+1) // epoch_time, "/", epoch_nbr)
        vect = som.unique_random_vector()
        som.train(i, epoch_time, vect)

    tile2.set_image(ImageQt(img.compress(som)))
    tile3.set_image(ImageQt(img.display_som(som.get_som_as_list())))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow("Tracking")
    # tile4 = Difference(window)
    tile3 = MapView(window)
    tile2 = ReconstructedImage(window)
    tile1 = Image(window)
    threading.Thread(target=run, name="run").start()
    sys.exit(app.exec_())
