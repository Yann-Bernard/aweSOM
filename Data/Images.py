from PIL import Image
from Models.SOM import *
from Models.Parameters import *
from dahuffman import HuffmanCodec
import os

# Choosing the number n of neurons or the side c of the prototypes. Si is the size of the initial image, h is the height of the image, w is the width, Cr is the compression ratio.
# n = (Si*Cr - 32*c²/(h*w))/8c²
# c² = Si*Cr/(8n+32/(h*w))

class Dataset:
    def __init__(self, path):
        self.nb_pictures = None
        self.data = None
        self.load_image(path)

    def load_image(self, path):
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

        print("\n" + path)
        print("Pictures number :", self.nb_pictures)
        if size[0] / pictures_dim[0] != self.nb_pictures[0] or size[0] / pictures_dim[0] != self.nb_pictures[0]:
            print("\tWarning - image size is not divisible by pictures dimensions, the result will be cropped")

    def compute_result(self, som):
        som_map = som.get_som_as_map()
        pixels = []
        winners = []
        for i in range(len(self.data)):
            w = som.winner(self.data[i])
            if w not in winners:
                winners.append(w)
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

        # n = neuron_nbr*neuron_nbr
        # print("Used neurons :", len(winners), "/", n, "(", len(winners)/n*100, "%)")

        return file

    def compression(self, som, name):
        file = self.compute_result(som)
        #file.show()
        file.save(output_path+name)

    def save_compressed(self, som, name):
        # winners = np.zeros((neuron_nbr, neuron_nbr))
        # for i in range(len(self.data)):
        #     w = som.winner(self.data[i]/255)
        #     winners[w] += 1
        winners = som.winners()
        file = open(output_path+name, "w")
        # file.write(str(som.get_som_as_list()))
        # res = ""
        # str_win = ""
        diff = Dataset.differential_coding(winners.flatten(), self.nb_pictures[1])

        # for i in range(len(self.data)):
        #     res += str(diff[i])+" "
        #     str_win += str(winners[i])+" "

        # # Codebook compression
        # codebook = som.get_som_as_list()
        # str_codebook = ""
        # for i in codebook:
        #     for j in range(len(i)):
        #         str_codebook += str(j)+" "
        #     str_codebook += "\n"

        codeNormal = HuffmanCodec.from_data(winners).encode(winners)
        codeDiff = HuffmanCodec.from_data(diff).encode(diff)
        hd = np.concatenate(som.get_som_as_list(), 0) * 255
        hd = np.array(hd, 'uint8')
        header = HuffmanCodec.from_data(hd).encode(hd)
        file.write(str(header))
        file.write(str(codeDiff))
        file.close()

        print("Taux de compression du codage différentiel :", len(codeNormal)/len(codeDiff))
        print("Taux de compression total :", len(self.data)*len(self.data[0])/(len(header)+len(codeDiff)))

    @staticmethod
    def compute_compression_ratio(data, som, winners, width):
        diff = Dataset.differential_coding(winners.flatten(), width)
        codeNormal = HuffmanCodec.from_data(winners).encode(winners)
        codeDiff = HuffmanCodec.from_data(diff).encode(diff)
        hd = np.concatenate(som.get_som_as_list(), 0) * 255
        hd = np.array(hd, 'uint8')
        header = HuffmanCodec.from_data(hd).encode(hd)
        return len(codeNormal)/len(codeDiff), len(data)*len(data[0])/(len(header)+len(codeDiff))

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
        data.extend(Dataset(path + f).data)
    return data

