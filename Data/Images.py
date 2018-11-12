from PIL import Image
from dahuffman import HuffmanCodec
from Parameters import *
import os


class ImageData:
    def __init__(self, path=input_path+image_name):
        self.nb_pictures = None
        self.data = None
        self.load(path)

    def load(self, path):
        self.data = []
        im = Image.open(path)
        size = np.flip(im.size, 0)  # For some strange reason the data isn't ordered in the same way as the size says
        px = np.array(im.getdata(), 'uint8')
        if len(px.shape) == 2:  # File has RGB colours
            px = np.hsplit(px, 3)[0]
        px = px.reshape(size)
        self.nb_pictures = np.array(np.divide(size, pictures_dim), dtype=int)
        px = px[0:self.nb_pictures[0] * pictures_dim[0], 0:self.nb_pictures[1] * pictures_dim[1]]  # Cropping the image to make it fit
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
        for i in range(len(self.data)):
            w = som.winner(self.data[i])
            pixels.append(som_map[w])
        px2 = []
        lst2 = ()
        for i in range(self.nb_pictures[0]):
            lst = ()
            for j in range(self.nb_pictures[1]):
                pixels[i*self.nb_pictures[1]+j] = pixels[i*self.nb_pictures[1]+j].reshape(pictures_dim)
                lst = lst + (pixels[i*self.nb_pictures[1]+j],)
            px2.append(np.concatenate(lst, axis=1))
            lst2 += (px2[i],)
        px = np.concatenate(lst2, axis=0)
        px *= 255
        px = np.array(px, 'uint8')
        file = Image.fromarray(px)
        return file

    @staticmethod
    def compute_compression_metrics(data, som, winners, width):
        diff = ImageData.differential_coding(winners.flatten(), width)
        normal_code = HuffmanCodec.from_data(winners).encode(winners)
        differential_code = HuffmanCodec.from_data(diff).encode(diff)
        hd = np.concatenate(som.get_som_as_list(), 0) * 255
        hd = np.array(hd, 'uint8')
        header = HuffmanCodec.from_data(hd).encode(hd)
        return len(normal_code)/len(differential_code), len(data)*len(data[0])/(len(header)+len(differential_code))

    @staticmethod
    def differential_coding(winners, width):
        diff = np.zeros(len(winners), dtype=int)
        # The first two lines are only using the previous element to differentiate
        diff[0] = winners[0]
        diff[2*width-1] = winners[2*width-1] - winners[width-1]
        for i in range(width-1):
            diff[i+1] = winners[i+1] - winners[i]
            diff[2*width-i-2] = winners[2*width-i-2] - winners[2*width-i-1]
        # Difference with the minimum gradient of 4 directions
        for i in range(2, int(len(winners)/width)):
            for j in range(width):
                left = np.inf
                top_left = np.inf
                top = np.abs(winners[(i-2)*width+j] - winners[(i-2)*width+j])
                top_right = np.inf
                if j > 1:
                    left = np.abs(winners[i*width+j-2] - winners[i*width+j-1])
                    top_left = np.abs(winners[(i-2)*width+j-2] - winners[(i-1)*width+j-1])
                if j < width-2:
                    top_right = np.abs(winners[(i-2)*width+j+2] - winners[(i-1)*width+j+1])
                min = np.min((left, top_left, top, top_right))
                if min == left:
                    diff[i*width+j] = winners[i*width+j] - winners[i*width+j-1]
                elif min == top_left:
                    diff[i*width+j] = winners[i*width+j] - winners[(i-2)*width+j]
                elif min == top:
                    diff[i*width+j] = winners[i*width+j] - winners[(i-2)*width+j]
                elif min == top_right:
                    diff[i*width+j] = winners[i*width+j] - winners[(i-2)*width+j]
        return diff

    def reverse_differential_coding(self, diff, width):
        winners = np.zeros(len(diff), dtype=int)
        # The first two lines are only using the previous element to differentiate
        winners[0] = diff[0]
        for i in range(width-1):
            winners[i+1] = winners[i] + diff[i+1]
        winners[2*width-1] = winners[width-1] + diff[2*width-1]
        for i in range(width-1):
            winners[2*width-i-2] = winners[2*width-i-1] + diff[2*width-i-2]
        # Difference with the minimum gradient of 4 directions
        for i in range(2, int(len(diff)/width)):
            for j in range(width):
                left = np.inf
                top_left = np.inf
                top = np.abs(winners[(i-2)*width+j] - winners[(i-2)*width+j])
                top_right = np.inf
                if j > 1:
                    left = np.abs(winners[i*width+j-2] - winners[i*width+j-1])
                    top_left = np.abs(winners[(i-2)*width+j-2] - winners[(i-1)*width+j-1])
                if j < width-2:
                    top_right = np.abs(winners[(i-2)*width+j+2] - winners[(i-1)*width+j+1])
                min = np.min((left, top_left, top, top_right))
                if min == left:
                    winners[i*width+j] = winners[i*width+j-1] + diff[i*width+j]
                elif min == top_left:
                    winners[i*width+j] = winners[(i-2)*width+j] + diff[i*width+j]
                elif min == top:
                    winners[i*width+j] = winners[(i-2)*width+j] + diff[i*width+j]
                elif min == top_right:
                    winners[i*width+j] = winners[(i-2)*width+j] + diff[i*width+j]
        return winners


def load_image_folder(path):
    files = os.listdir(path)
    data = []
    for f in files:
        data.extend(ImageData(path + f).data)
    return data

