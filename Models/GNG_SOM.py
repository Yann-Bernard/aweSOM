from Models.DynamicSOM import *

# A Growing Neural Gases variant of DynamicSOM.
# Doesn't work well.


class GNG_SOM(DynamicSOM):
    def __init__(self, data, connexion_matrices, threshold=switch_threshold, eps_s=epsilon_start, eps_e=epsilon_end, sig_s=sigma_start, sig_e=sigma_end, ep_nb=epoch_nbr):
        self.epsilon = eps_s
        self.epsilon_stepping = (eps_e - eps_s) / ep_nb

        self.sigma = sig_s
        self.sigma_stepping = (sig_e - sig_s) / ep_nb

        super().__init__(data, connexion_matrices, threshold, eps_s, eps_e, sig_s, sig_e, ep_nb)

    def winner(self, vector, distance=dist_quad):
        dist = np.empty_like(self.nodes, dtype=float)
        for x in range(neuron_nbr):  # Computes the distances between the tested vector and all nodes
            for y in range(neuron_nbr):
                self.nodes[x, y].t += 1
                dist[x, y] = distance(self.nodes[x, y].weight, vector)
        s1 = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
        dist[s1] = np.inf
        s2 = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
        return s1, s2  # Returning the Best Matching Unit's index.

    def train(self, iteration, epoch_time, vector_coordinates, f=normalized_gaussian, distance=dist_quad):
        if iteration % epoch_time == 0:
            self.epsilon += self.epsilon_stepping
            self.sigma += self.sigma_stepping
            if dsom and iteration > 0:
                self.check_allegiance()
            self.refresh_distance_vector = True

        if self.refresh_distance_vector:
            for i in range(len(self.distance_vector)):
                self.distance_vector[i] = f(i/(len(self.distance_vector)-1), self.sigma)
            if log_gaussian_vector:
                print(self.distance_vector) 

        vector = self.data[vector_coordinates]

        # Getting the Best matching unit
        s1, s2 = self.winner(vector, distance)
        self.nodes[s1].nb_bmu += 1
        self.nodes[s1].error += np.sum(np.abs(vector-self.nodes[s1].weight))

        self.nodes[s1].t = 1
        self.nodes[s2].t = 1
        self.updating_weights(s1, vector)

        return s1[0], s1[1]

    def updating_weights(self, bmu, vector):
        for x in range(neuron_nbr):  # Updating weights of all nodes
            for y in range(neuron_nbr):
                dist = self.neural_dist[bmu[1]*neuron_nbr+bmu[0], y*neuron_nbr+x]
                if dist == 0:  # The BMU
                    self.nodes[x, y].weight += epsilon_b * (vector-self.nodes[x, y].weight)
                if dist == 1:  # The closest of the BMU
                    self.nodes[x, y].weight += epsilon_n * (vector-self.nodes[x, y].weight)
