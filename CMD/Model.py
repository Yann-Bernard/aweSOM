from Data.Images import *
from Models.SOM import *
from Models.Connections import *
from Models.DynamicSOM import *


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
    img = Dataset(path)
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


def compute_mean_error(datacomp, datamat, SOMList):
    error = np.zeros(len(datacomp))
    for i in range(len(datacomp)):
        error[i] = np.mean(np.abs(datamat[i] - SOMList[datacomp[i]]))*255
    return np.mean(error)


def peak_signal_to_noise_ratio(datacomp, datamat, SOMList):
    error = np.zeros(len(datacomp))
    for i in range(len(datacomp)):
        error[i] = np.mean((datamat[i] - SOMList[datacomp[i]])**2)
    return 10*np.log10(1/np.mean(error))


def run():
    np.random.seed(1024)
    img = Dataset("./image/limited_test/peppers.pgm")
    data = img.data
    # data = load_image_folder("./image/")

    datacomp = np.zeros(len(data), int)  # datacomp est la liste du numero du neurone vainqueur pour l'imagette correspondante
    old = np.array(datacomp)

    epoch_time = len(data)
    nb_iter = epoch_time * epoch_nbr

    carte = DynamicSOM(data, star())
#    datacomp = carte.winners()

#    print("Initial mean pixel error SOM: ", compute_mean_error(datacomp, data, carte.get_som_as_list()))
#    print("Initial PSNR: ", peak_signal_to_noise_ratio(datacomp, data, carte.get_som_as_list()))

    for i in range(nb_iter):
        # The training vector is chosen randomly
        if i % epoch_time == 0:
             carte.generate_random_list()
        vect = carte.unique_random_vector()

        carte.train(i, epoch_time, vect)
        if (i+1) % epoch_time == 0:
            print("Epoch : ", (i+1) // epoch_time, "/", epoch_nbr)
            if log_execution:
                datacomp = carte.winners()
                diff = np.count_nonzero(datacomp - old)
                print("Changed values SOM :", diff)
                print("Mean pixel error SOM: ", compute_mean_error(datacomp, data, carte.get_som_as_list()))
                print("PSNR: ", peak_signal_to_noise_ratio(datacomp, data, carte.get_som_as_list()))
                old = np.array(datacomp)

    carte.print_connexions()
    datacomp = carte.winners()
    print(datacomp)
    print("Final mean pixel error SOM: ", compute_mean_error(datacomp, data, carte.get_som_as_list()))
    print("Final PSNR: ", peak_signal_to_noise_ratio(datacomp, data, carte.get_som_as_list()))

    img.compression(carte, "star_"+str(neuron_nbr) + "n_"+str(pictures_dim[0])+"x"+str(pictures_dim[1])+"_"+str(epoch_nbr)+"epoch_comp.png")
    img.save_compressed(carte, "star_compressed.som")
    if psom:
        im1 = display_som_links(carte.get_som_as_list(), carte.neural_adjacency_matrix)
        im1.save(output_path+"links.png")
    im2 = display_som(carte.get_som_as_list())
    im2.save(output_path + "star_"+str(neuron_nbr) + "n_" + str(pictures_dim[0])+"x"+str(pictures_dim[1])+"_"+str(epoch_nbr)+"epoch_carte.png")


def run_from_som():
    img = Dataset("./image/Audrey.png")
    data = img.data
    carte = SOM(data, kohonen())
    load_som_as_image("./results/deep/star_12n_3x3_500epoch_comp.png", carte)
    img.compression(carte, "reconstruction_500epoch.png")
    im2 = display_som(carte.get_som_as_list())
    im2.save(output_path + "som_500epoch.png")
