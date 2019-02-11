import threading

import numpy as np
import sys

from PIL import ImageChops
from PIL.ImageQt import ImageQt
from PyQt5.QtWidgets import QApplication

from Data.ColoredImage import ColoredImage
from Data.Images import ImageData
from CMD.Single_Run import display_som
from GUI.View.DistanceMapView import DistanceMapView
from GUI.View.Image import Image
from GUI.View.MainWindow import MainWindow
from GUI.View.MapView import MapView
from GUI.View.Module import Module
from GUI.View.ReconstructedImage import ReconstructedImage
from Models.NPSOM.Connections import kohonen, star
from Models.SOM import SOM
from Parameters import *


def run(channel=None):
    img = ColoredImage(channel=channel)
    np.random.seed(1024)
    data = img.data
    epoch_time = len(data)
    nb_iter = epoch_time * epoch_nbr
    som = SOM(data, kohonen())

    for i in range(nb_iter):
        # The training vector is chosen randomly
        if i % epoch_time == 0:
            som.generate_random_list()
            if channel == 0:
                tile4.set_image(ImageQt(img.compress_channel(som,0)))
            elif channel == 1:
                tile5.set_image(ImageQt(img.compress_channel(som,1)))
            elif channel == 2:
                tile6.set_image(ImageQt(img.compress_channel(som,2)))
            #tile3.set_image(ImageQt(img.display_som_channel(som.get_som_as_list())))
            print("Epoch : ", (i+1) // epoch_time, "/", epoch_nbr)
        vect = som.unique_random_vector()
        som.train(i, epoch_time, vect)

    reconstructed_image = img.compress(som)
    tile2.set_image(ImageQt(reconstructed_image))
    reconstructed_image.save(output_path + "koh_"+str(neuron_nbr) + "n_" + str(pictures_dim[0])+"x"+str(pictures_dim[1])+"_"+str(epoch_nbr)+"epoch_image.png")

    som_as_image = img.display_som(som.get_som_as_list())
    tile3.set_image(ImageQt(som_as_image))
    som_as_image.save(output_path + "koh_"+str(neuron_nbr) + "n_" + str(pictures_dim[0])+"x"+str(pictures_dim[1])+"_"+str(epoch_nbr)+"epoch_map.png")
    print("Finished")
    input("press enter to start tracking")
    track(som)


def track(som):
    max = 101
    for i in range(42, max):
        path = "./Data/images/tracking/sailboat00"+"{0:0=3d}".format(i)+".png"
        current = ColoredImage(path)
        img_compressed = current.compress(som)
        out = ImageChops.difference(current.im, img_compressed)
        out = out.convert("L")
        tile1.set_image(ImageQt(current.im))
        tile2.set_image(ImageQt(img_compressed))
        tile4.set_image(ImageQt(out))
        if i == 45:
            img_compressed.save(output_path+"comp.png")
            out.save(output_path+"out.png")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow("Tracking")
    tile6 = MapView(window)
    tile5 = MapView(window)
    tile4 = MapView(window)
    tile3 = DistanceMapView(window)
    tile2 = ReconstructedImage(window)
    tile1 = Image(window)
    threading.Thread(target=run, name="red", args=(0,)).start()
    threading.Thread(target=run, name="green", args=(1,)).start()
    threading.Thread(target=run, name="blue", args=(2,)).start()
    sys.exit(app.exec_())
