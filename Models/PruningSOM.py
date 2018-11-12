from Models.SOM import *

# SOM with pruning of edges when two neighbouring vertices are too far away.
# With linearly decreasing parameters and cutting threshold.


class PruningSOM(SOM):
    def __init__(self, data, connexion_matrices, eps_s=epsilon_start, eps_e=epsilon_end, sig_s=sigma_start,
                 sig_e=sigma_end, ep_nb=epoch_nbr):
        super().__init__(data, connexion_matrices, eps_s, eps_e, sig_s, sig_e, ep_nb)

    def train(self, iteration, epoch_time, vector_coordinates, f=normalized_gaussian, distance=dist_quad):
        if iteration % epoch_time == 0:
            self.epsilon += self.epsilon_stepping
            self.sigma += self.sigma_stepping
            if iteration > 0:
                self.pruning_neighbors()
            self.refresh_distance_vector = True

        if self.refresh_distance_vector:
            for i in range(len(self.distance_vector)):
                self.distance_vector[i] = f(i / (len(self.distance_vector) - 1), self.sigma)
            if log_gaussian_vector:
                print(self.distance_vector)

        vector = self.data[vector_coordinates]

        # Getting the Best matching unit
        bmu = self.winner(vector, distance)
        self.nodes[bmu].t = 1
        self.updating_weights(bmu, vector)

        return bmu[0], bmu[1]

    def pruning_neighbors(self):
        for x in range(neuron_nbr - 1):
            for y in range(neuron_nbr - 1):
                self.pruning_check(x, y, x + 1, y)
                self.pruning_check(x, y, x, y + 1)
        self.compute_neurons_distance()

    def pruning_check(self, x1, y1, x2, y2):
        one = y1 * neuron_nbr + x1
        two = y2 * neuron_nbr + x2
        if self.neural_adjacency_matrix[one, two] != 0 and self.neural_adjacency_matrix[one, two] != np.inf:
            diff = manhattan_dist(self.nodes[x1, y1].weight, self.nodes[x2, y2].weight)
            probability = np.exp(-1 / omega * 1 / (diff * self.nodes[x1, y1].t * self.nodes[x2, y2].t))
            if np.random.rand() < probability:
                print("Removed (", x1, ",", y1, ") - (", x2, ",", y2, ") probability : ", probability)
                self.remove_edges((x1, y1), (x2, y2))