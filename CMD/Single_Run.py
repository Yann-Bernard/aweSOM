from Data.Images import *
from Models.SOM import *
from Models.NPSOM.Connections import *
from Models.DynamicSOM import *
from Models.PCSOM import *


def noLink():
    pix = np.full(pictures_dim,255)
    return pix


def hLink():
    pix = np.full(pictures_dim,255)
    mid = pictures_dim[0]//2
    for j in range(pictures_dim[1]):
        pix[mid][j]=0
    return pix


def vLink():
    pix = np.full(pictures_dim,255)
    mid = pictures_dim[1]//2
    for i in range(pictures_dim[0]):
        pix[i][mid]=0
    return pix


def display_som(som_list):
    som_list = som_list*255
    px2 = []
    lst2 = ()
    for y in range(neuron_nbr):
        lst = ()
        for x in range(neuron_nbr):
            som_list[y * neuron_nbr + x] = som_list[y * neuron_nbr + x].reshape(pictures_dim)
            lst = lst + (som_list[y * neuron_nbr + x],)
        px2.append(np.concatenate(lst, axis=1))
        lst2 += (px2[y],)
    px = np.concatenate(lst2, axis=0)
    px = np.array(px, 'uint8')

    som_image = Image.fromarray(px)
    #  som_image.show()
    return som_image


def load_som_as_image(path, som):
    img = ImageData(path)
    som.set_som_as_list(img.data)


def display_som_links(som_list, adj):
    #px2 = []
    lst2 = ()
    for y in range(neuron_nbr):
        lst = ()
        for x in range(neuron_nbr):
            som_list[y * neuron_nbr + x] = som_list[y * neuron_nbr + x].reshape(pictures_dim)
            lst = lst + (som_list[y * neuron_nbr + x],)
            if x < neuron_nbr-1:
                if adj[y*neuron_nbr+x][y*neuron_nbr+x+1] == 0:
                    lst = lst + (noLink(),)
                else:
                    lst = lst + (hLink(),)
        #px2.append(np.concatenate(lst, axis=1))
        lst2 += (np.concatenate(lst, axis=1),)
        lst = ()
        if y < neuron_nbr-1:
            for j in range(neuron_nbr):
                if adj[y*neuron_nbr+x][(y+1)*neuron_nbr+j] == 0:
                    lst = lst + (noLink(),)
                else:
                    lst = lst + (vLink(),)
                if x < neuron_nbr-1:
                    lst = lst + (noLink(),)
            #px2.append(np.concatenate(lst, axis=1))
            lst2 += (np.concatenate(lst, axis=1),)
    px = np.concatenate(lst2, axis=0)
    px = np.array(px, 'uint8')

    som_image = Image.fromarray(px)
    return som_image


def run():
    np.random.seed(1024)
    img = ImageData()
    data = img.data
    # data = load_image_folder("./images/")

    winners_list = np.zeros(len(data), int)  # list of BMU for each corresponding training vector
    old_winners = np.array(winners_list)

    epoch_time = len(data)
    nb_iter = epoch_time * epoch_nbr

    som = DynamicSOM(data, star())

    for i in range(nb_iter):
        # The training vector is chosen randomly
        if i % epoch_time == 0:
             som.generate_random_list()
        vect = som.unique_random_vector()

        som.train(i, epoch_time, vect)
        if (i+1) % epoch_time == 0:
            print("Epoch : ", (i+1) // epoch_time, "/", epoch_nbr)
            if log_execution:
                winners_list = som.get_all_winners()
                diff = np.count_nonzero(winners_list - old_winners)
                print("Changed values:", diff)
                print("Mean pixel error SOM estimation: ", som.compute_mean_error(winners_list))
                print("PSNR estimation: ", som.peak_signal_to_noise_ratio(winners_list))
                old_winners = np.array(winners_list)

    # som.print_connexions()
    winners_list = som.get_all_winners()
    print(winners_list)
    print("Final mean pixel error SOM: ", som.compute_mean_error(winners_list))
    print("Final PSNR: ", som.peak_signal_to_noise_ratio(winners_list))

    compressed_image = img.compress(som)
    compressed_image.save("star_"+str(neuron_nbr) + "n_"+str(pictures_dim[0])+"x"+str(pictures_dim[1])+"_"+str(epoch_nbr)+"epoch_comp.png")
    som_as_image = display_som(som.get_som_as_list())
    som_as_image.save(output_path + "star_"+str(neuron_nbr) + "n_" + str(pictures_dim[0])+"x"+str(pictures_dim[1])+"_"+str(epoch_nbr)+"epoch_carte.png")


def run_from_som():
    img = Dataset("./images/Audrey.png")
    data = img.data
    carte = SOM(data, kohonen())
    load_som_as_image("./results/deep/star_12n_3x3_500epoch_comp.png", carte)
    img.compression(carte, "reconstruction_500epoch.png")
    im2 = display_som(carte.get_som_as_list())
    im2.save(output_path + "som_500epoch.png")


if __name__ == '__main__':
    run()
