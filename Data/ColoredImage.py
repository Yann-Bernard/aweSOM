from PIL import Image
from dahuffman import HuffmanCodec
from Parameters import *
import os


class ColoredImage:
    def __init__(self, path=input_path+image_name):
        self.im = None
        self.nb_pictures = None
        self.real_picture_dim = None
        self.data = None
        print(path)
        self.load(path)

    def load(self, path):
        self.data = []
        self.im = Image.open(path)
        size = np.flip(self.im.size, 0)  # For some strange reason the data isn't ordered in the same way as the size says
        size[1] = size[1]*3
        px = np.array(self.im.getdata(), 'uint8')
        print(size)
        px = px.reshape(size)
        self.real_picture_dim = [pictures_dim[0], pictures_dim[1]*3]
        self.nb_pictures = np.array(np.divide(size, self.real_picture_dim), dtype=int)
        px = px[0:self.nb_pictures[0] * self.real_picture_dim[0], 0:self.nb_pictures[1] * self.real_picture_dim[1]]  # Cropping the image to make it fit
        px = np.vsplit(px, self.nb_pictures[0])
        for i in px:
            j = np.hsplit(i, self.nb_pictures[1])
            for picture in j:
                self.data.append(picture.flatten())
        self.data = np.array(self.data)/255

        if log_data_load:
            print("\n" + path)
            print("Pictures number :", self.nb_pictures)
            if size[0] / pictures_dim[0] != self.nb_pictures[0] or size[0] / pictures_dim[0] != self.nb_pictures[0]:
                print("\tWarning - image size is not divisible by pictures dimensions, the result will be cropped")

    def compress(self, som):
        som_map = som.get_som_as_map()
        pixels = []
        dim = [self.real_picture_dim[0], self.real_picture_dim[1]//3, 3]
        for i in range(len(self.data)):
            w = som.winner(self.data[i])
            pixels.append(som_map[w])
        px2 = []
        lst2 = ()
        for i in range(self.nb_pictures[0]):
            lst = ()
            for j in range(self.nb_pictures[1]):
                pixels[i*self.nb_pictures[1]+j] = pixels[i*self.nb_pictures[1]+j].reshape(dim)
                lst = lst + (pixels[i*self.nb_pictures[1]+j],)
            px2.append(np.concatenate(lst, axis=1))
            lst2 += (px2[i],)
        px = np.concatenate(lst2, axis=0)
        px *= 255
        px = np.array(px, 'uint8')
        file = Image.fromarray(px)
        return file

    def display_som(self, som_list):
        som_list = som_list * 255
        px2 = []
        lst2 = ()
        dim = [self.real_picture_dim[0], self.real_picture_dim[1]//3, 3]
        for y in range(neuron_nbr):
            lst = ()
            for x in range(neuron_nbr):
                som_list[y * neuron_nbr + x] = som_list[y * neuron_nbr + x].reshape(dim)
                lst = lst + (som_list[y * neuron_nbr + x],)
            px2.append(np.concatenate(lst, axis=1))
            lst2 += (px2[y],)
        px = np.concatenate(lst2, axis=0)
        px = np.array(px, 'uint8')

        som_image = Image.fromarray(px)
        #  som_image.show()
        return som_image
