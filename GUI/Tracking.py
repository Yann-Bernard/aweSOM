import PIL
import threading

import numpy as np
import sys

from PIL import ImageChops
from PIL.ImageQt import ImageQt
from PyQt5.QtWidgets import QApplication

from Data.ColoredImage import ColoredImage
from Data.Images import ImageData
from CMD.Single_Run import display_som
from GUI.DNF import DNF
from GUI.View.DNFView import DNFView
from GUI.View.DistanceMapView import DistanceMapView
from GUI.View.Image import Image
from GUI.View.MainWindow import MainWindow
from GUI.View.MapView import MapView
from GUI.View.Module import Module
from GUI.View.ReconstructedImage import ReconstructedImage
from Models.NPSOM.Connections import kohonen, star
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
    max = 250
    for i in range(24, max):
        path = "./Data/images/tracking/ducks/ducks"+"{0:0=5d}".format(i)+".png"
        current = ColoredImage(path)
        img_compressed = current.compress(som)
        out = ImageChops.difference(current.im, img_compressed)
        out = out.convert("L")
        tile1.set_image(ImageQt(current.im))
        tile2.set_image(ImageQt(img_compressed))
        tile4.set_image(ImageQt(out))
        dnf_update(out)
        out.save(output_path+"out.png")

def dnf_update(saliency):
    inp = np.array(np.asarray(saliency))
    dnf = DNF(np.size(inp,0), np.size(inp,1))
    inp = np.divide(inp, np.max(inp))
    dnf.input = inp
    dnf_out = None
    for i in range(10):
        dnf.update_map()
        px = dnf.potentials*255
        px = np.array(px, 'uint8')
        dnf_out = PIL.Image.fromarray(px)
        tile5.set_image(ImageQt(dnf_out))
    dnf_out.save(output_path + "dnf_out.png")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow("Tracking")
    tile5 = DNFView(window)
    tile4 = DistanceMapView(window)
    tile3 = MapView(window)
    tile2 = ReconstructedImage(window)
    tile1 = Image(window)
    threading.Thread(target=run, name="run").start()
    sys.exit(app.exec_())
