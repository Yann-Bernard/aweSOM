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
    imglist = (ColoredImage(channel=0), ColoredImage(channel=1), ColoredImage(channel=2))
    np.random.seed(1024)
    data = imglist[0].data
    epoch_time = len(data)
    nb_iter = epoch_time * epoch_nbr
    somlist = (SOM(imglist[0].data, kohonen()), SOM(imglist[1].data, kohonen()), SOM(imglist[2].data, kohonen()))
    for i in range(nb_iter):
        # The training vector is chosen randomly
        if i % epoch_time == 0:
            somlist[0].generate_random_list()
            tile4.set_image(ImageQt(imglist[0].display_som_channel(somlist[0].get_som_as_list(), 0)))
            tile5.set_image(ImageQt(imglist[1].display_som_channel(somlist[1].get_som_as_list(), 1)))
            tile6.set_image(ImageQt(imglist[2].display_som_channel(somlist[2].get_som_as_list(), 2)))
            reconstructed_image = compress_all(somlist, imglist)
            tile2.set_image(ImageQt(reconstructed_image))
            print("Epoch : ", (i+1) // epoch_time, "/", epoch_nbr)
        vect = somlist[0].unique_random_vector()
        for s in somlist:
            s.train(i, epoch_time, vect)
    reconstructed_image = compress_all(somlist, imglist)
    tile2.set_image(ImageQt(reconstructed_image))
    reconstructed_image.save(output_path + "koh_"+str(neuron_nbr) + "n_" + str(pictures_dim[0])+"x"+str(pictures_dim[1])+"_"+str(epoch_nbr)+"epoch_image.png")

    tile4.set_image(ImageQt(imglist[0].display_som_channel(somlist[0].get_som_as_list(), 0)))
    tile5.set_image(ImageQt(imglist[1].display_som_channel(somlist[1].get_som_as_list(), 1)))
    tile6.set_image(ImageQt(imglist[2].display_som_channel(somlist[2].get_som_as_list(), 2)))
    #som_as_image.save(output_path + "koh_"+str(neuron_nbr) + "n_" + str(pictures_dim[0])+"x"+str(pictures_dim[1])+"_"+str(epoch_nbr)+"epoch_map.png")
    print("Finished")
    input("press enter to start tracking")
    track(somlist)



def compress_all(somlist, imglist):
    complist = ()
    for i in range(3):
        complist = complist + (imglist[i].compress_channel(somlist[i], i),)
    return PIL.Image.merge("RGB",complist)

def track(somlist):
    max = 250
    for i in range(24, max):
        path = "./Data/images/tracking/ducks/ducks"+"{0:0=5d}".format(i)+".png"
        current = ColoredImage(path)
        imglist = (ColoredImage(path, channel=0), ColoredImage(path, channel=1), ColoredImage(path, channel=2))
        img_compressed = compress_all(somlist, imglist)
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
    tile7 = DNFView(window)
    tile6 = MapView(window)
    tile5 = MapView(window)
    tile4 = MapView(window)
    tile3 = DistanceMapView(window)
    tile2 = ReconstructedImage(window)
    tile1 = Image(window)
    threading.Thread(target=run, name="run").start()
    sys.exit(app.exec_())
