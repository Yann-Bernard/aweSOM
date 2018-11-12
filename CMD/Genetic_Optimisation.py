from CMD.Model import *
from Models.DynamicSOM import *
import itertools
import multiprocessing as mp


def mutate(value, bounds):
    value += (np.random.rand() * 2 - 1) * mutation_value * (bounds[1] - bounds[0])
    if value > bounds[1]:
        value = bounds[1]
    elif value < bounds[0]:
        value = bounds[0]
    return value


class DynamicGenome:
    def __init__(self):
        self.threshold = (range_threshold[1]-range_threshold[0]) * np.random.random() + range_threshold[0]
        self.fitness = 255

    def crossover(self, father, mother):
        ratio = np.random.random()
        self.threshold = father.threshold * ratio + (1 - ratio) * mother.threshold
        self.fitness = 255

    def mutation(self):
        if probability_mutation > np.random.random():
            self.threshold = mutate(self.threshold, range_threshold)

    def run_fitness(self, data):
        epoch_time = len(data)
        nb_iter = epoch_time * epoch_nbr
        som = DynamicSOM(data, star(), self.threshold)
        for i in range(nb_iter):
            if i % epoch_time == 0:
                som.generate_random_list()
            vector = som.unique_random_vector()
            som.train(i, epoch_time, vector)
        data_comp = som.winners()
        self.fitness = peak_signal_to_noise_ratio(data_comp, data, som.get_som_as_list())
        return self

    def to_string(self):
        res = ""
        res += "Fitness :"+str(self.fitness)
        res += " (threshold: " + str(self.threshold)+")"
        return res

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __eq__(self, other):
        return self.fitness == other.fitness


class Genome:
    def __init__(self):
        self.epoch_nbr = np.random.random_integers(range_epoch_nbr[0], range_epoch_nbr[1])
        self.epsilon_start = (range_epsilon_start[1]-range_epsilon_start[0]) * np.random.random() + range_epsilon_start[0]
        self.epsilon_end = (range_epsilon_end[1]-range_epsilon_end[0]) * np.random.random() + range_epsilon_end[0]
        self.sigma_start = (range_sigma_start[1]-range_sigma_start[0]) * np.random.random() + range_sigma_start[0]
        self.sigma_end = (range_sigma_end[1]-range_sigma_end[0]) * np.random.random() + range_sigma_end[0]
        self.fitness = 255

    def crossover(self, father, mother):
        ratio = np.random.random()
        self.epoch_nbr = int(np.round(father.epoch_nbr * ratio + (1 - ratio) * mother.epoch_nbr))
        self.epsilon_start = father.epsilon_start * ratio + (1 - ratio) * mother.epsilon_start
        self.epsilon_end = father.epsilon_end * ratio + (1 - ratio) * mother.epsilon_end
        self.sigma_start = father.sigma_start * ratio + (1 - ratio) * mother.sigma_start
        self.sigma_end = father.sigma_end * ratio + (1 - ratio) * mother.sigma_end
        self.fitness = 255

    def mutation(self):
        if probability_mutation > np.random.random():
            self.epoch_nbr = int(mutate(self.epoch_nbr, range_epoch_nbr))
        if probability_mutation > np.random.random():
            self.epsilon_start = mutate(self.epsilon_start, range_epsilon_start)
        if probability_mutation > np.random.random():
            self.epsilon_end = mutate(self.epsilon_end, range_epsilon_end)
        if probability_mutation > np.random.random():
            self.sigma_start = mutate(self.sigma_start, range_sigma_start)
        if probability_mutation > np.random.random():
            self.sigma_end = mutate(self.sigma_end, range_sigma_end)

    def run_fitness(self, data):
        epoch_time = len(data)
        nb_iter = epoch_time * self.epoch_nbr
        som = SOM(data, star(), self.epsilon_start, self.epsilon_end, self.sigma_start, self.sigma_end, self.epoch_nbr)
        for i in range(nb_iter):
            if i % epoch_time == 0:
                som.generate_random_list()
            vector = som.unique_random_vector()
            som.train(i, epoch_time, vector)
        data_comp = som.winners()
        self.fitness = peak_signal_to_noise_ratio(data_comp, data, som.get_som_as_list())
        return self

    def to_string(self):
        res = ""
        res += "Fitness :"+str(self.fitness)
        res += " in " + str(self.epoch_nbr)+" epochs"
        res += "(eps_s: " + str(self.epsilon_start) + " eps_e: " + str(self.epsilon_end) \
               + " sig_s: " + str(self.sigma_start) + " sig_e: " + str(self.sigma_end) + ")"
        return res

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __eq__(self, other):
        return self.fitness == other.fitness


class ConnexionGenome:
    def __init__(self):
        self.connexion_matrix = np.empty((neuron_nbr, neuron_nbr, 5, 5))
        for y in range(neuron_nbr):
            for x in range(neuron_nbr):
                matrix = np.zeros((5, 5), dtype=int)
                for i in range(5):
                    for j in range(5):
                        if i == j:
                            matrix[i, j] = 0
                        elif i == 4 or j == 4:
                            matrix[i, j] = 1 if np.random.random() < probability_neural_link else 0
                        else:
                            matrix[i, j] = 1 if np.random.random() < probability_link else 0
                self.connexion_matrix[x, y] = matrix
        self.connexion_matrix = kohonen()
        self.fitness = 255

    def crossover(self, father, mother):
        horizontal_cut = np.random.random() < 0.5
        cut = np.random.random()*neuron_nbr
        for y in range(neuron_nbr):
            for x in range(neuron_nbr):
                if (horizontal_cut and x < cut) or (not horizontal_cut and y < cut):
                    self.connexion_matrix[x, y] = father.connexion_matrix[x, y]
                else:
                    self.connexion_matrix[x, y] = mother.connexion_matrix[x, y]
        self.fitness = 255

    def mutation(self):
        for y in range(neuron_nbr):
            for x in range(neuron_nbr):
                if np.random.random() < probability_mutation:
                    matrix = np.zeros((5, 5), dtype=int)
                    for i in range(5):
                        for j in range(5):
                            if i != j:
                                matrix[i, j] = matrix[i, j] if np.random.random() > probability_mutation else 1 - matrix[i, j]
                    self.connexion_matrix[x, y] = matrix
        self.fitness = 255

    def run_fitness(self, data):
        epoch_time = len(data)
        nb_iter = epoch_time * epoch_nbr
        som = SOM(data, self.connexion_matrix)
        for i in range(nb_iter):
            if i % epoch_time == 0:
                som.generate_random_list()
            vector = som.unique_random_vector()
            som.train(i, epoch_time, vector)
        data_comp = som.winners()
        self.fitness = peak_signal_to_noise_ratio(data_comp, data, som.get_som_as_list())
        return self

    def to_string(self):
        res = ""
        res += "Fitness :"+str(self.fitness)
        return res

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __eq__(self, other):
        return self.fitness == other.fitness


class Population:
    def __init__(self):
        img = Dataset("./image/limited_test/peppers.pgm")
        self.data = img.data
        self.current = []
        for i in range(nb_individuals):
            self.current.append(DynamicGenome())

    def run(self):
        for i in range(nb_generations):
            print("Generation "+str(i)+"/"+str(nb_generations))
            self.evaluate_all()
            self.select()
        print(self.current[0].to_string())

    def evaluate_all(self):
        pool = mp.Pool(nb_individuals)
        self.current = pool.starmap(DynamicGenome.run_fitness, zip(self.current, itertools.repeat(self.data)))
        pool.close()
        pool.join()

    def select(self):
        self.current.sort(reverse=True)
        for j in self.current:
            print(j.to_string())
        new = []
        for i in range(nb_individuals):
            if i <= elite_proportion*nb_individuals:
                new.append(self.current[i])
            else:
                child = DynamicGenome()
                child.crossover(self.current[np.random.randint(0, nb_individuals)], self.current[np.random.randint(0, nb_individuals)])
                child.mutation()
                new.append(child)
        self.current = new




a = Population()
a.run()
