from CMD.Single_Run import *
import multiprocessing as mp
from Models.DynamicSOM import *

connexions_matrices = {"koh": kohonen(), "sw": small_worlds(), "star": star()}
images = {}
database = {}

class Run:
    def __init__(self, eS, eE, sS, sE, connexion):
        self.conn = connexion
        self.epoch_nbr = epoch_nbr
        self.epsilon_start = eS
        self.epsilon_end = eE
        self.sigma_start = sS
        self.sigma_end = sE
        self.mean = -1
        self.psnr = -1
        self.diff = -1
        self.comp = -1

    def run_fitness(self, data, width):
        epoch_time = len(data)
        nb_iter = epoch_time * self.epoch_nbr

        som = SOM(data, connexions_matrices[self.conn], self.epsilon_start, self.epsilon_end, self.sigma_start, self.sigma_end, self.epoch_nbr)
        for i in range(nb_iter):
            if i % epoch_time == 0:
                som.generate_random_list()
            vector = som.unique_random_vector()
            som.train(i, epoch_time, vector)
        data_comp = som.winners()
        self.mean = compute_mean_error(data_comp, data, som.get_som_as_list())
        self.psnr = peak_signal_to_noise_ratio(data_comp, data, som.get_som_as_list())
        self.diff, self.comp = Dataset.compute_compression_ratio(data, som, data_comp, width)
        print(self.to_string())

    def to_string(self):
        res = self.conn+";"
        res += str(self.epsilon_start)+";"+str(self.epsilon_end)+";"
        res += str(self.sigma_start)+";"+str(self.sigma_end)+";"
        res += str(self.mean)+";"+str(self.psnr)+";"
        res += str(self.diff)+";"+str(self.comp)
        return res


class StatsRun:
    def __init__(self, connexion, img, seed_one, seed_two):
        self.conn = connexion
        self.epoch_nbr = epoch_nbr
        self.seed_one = seed_one
        self.seed_two = seed_two
        self.img = img
        self.mean = -1
        self.psnr = -1
        self.diff = -1
        self.comp = -1

    def run_fitness(self):
        data = database[self.img]
        epoch_time = len(data)
        nb_iter = epoch_time * self.epoch_nbr
        np.random.seed(self.seed_one)
        som = DynamicSOM(data, connexions_matrices[self.conn])
        np.random.seed(self.seed_two)
        for i in range(nb_iter):
            if i % epoch_time == 0:
                som.generate_random_list()
            vector = som.unique_random_vector()
            som.train(i, epoch_time, vector)
        data_comp = som.winners()
        self.mean = compute_mean_error(data_comp, data, som.get_som_as_list())
        self.psnr = peak_signal_to_noise_ratio(data_comp, data, som.get_som_as_list())
        self.diff, self.comp = Dataset.compute_compression_ratio(data, som, data_comp, images[self.img].nb_pictures[1])
        self.changed_conn = som.changed_connexions
        # images[self.img].compression(som, self.img+"_"+self.conn+"_" + str(neuron_nbr) + "n_" + str(pictures_dim[0]) + "x" + str(pictures_dim[1]) + "_comp.png")
        # images[self.img].save_compressed(som, self.conn+"_compressed.som")
        # im2 = display_som(som.get_som_as_list())
        # im2.save(output_path + self.img+"_"+self.conn+"_" + str(neuron_nbr) + "n_" + str(pictures_dim[0]) + "x" + str(pictures_dim[1]) + "_carte.png")

        print(self.to_string())
        #print(np.bincount(data_comp))

    def to_string(self):
        res = self.conn+";"+self.img+";"
        res += str(pictures_dim[0])+";"+str(neuron_nbr)+";"
        res += str(self.mean)+";"+str(self.psnr)+";"
        res += str(self.diff)+";"+str(self.comp)+";"+str(self.changed_conn)
        return res


class FullTest:
    def __init__(self):
        self.img_str = os.listdir(input_path)
        for f in self.img_str:
            images[f] = Dataset(input_path + f)
            database[f] = images[f].data
        self.current = []
        connex = ("star",)
        for i in connex:
            for j in images:
                for k in range(0, 10):
                    for l in range(0, 10):
                        self.current.append(StatsRun(i, j, k, l))

    def run(self):
        pool = mp.Pool(8)
        self.current = pool.starmap(StatsRun.run_fitness, zip(self.current))
        pool.close()
        pool.join()

a = FullTest()
a.run()
