from Models.DynamicSOM import *

# A DynamicSOM and PCSOM fusion.
# Without pruning, only fast learning.


class DPCNeuron(DynamicNeuron):
    def __init__(self, x, y, shape, min, max, connections, neighbour):
        super().__init__(x, y, shape, min, max, connections, neighbour)
        self.connected_list = []


class DynamicPCSOM(DynamicSOM):
    def __init__(self, data, connexion_matrices, threshold=switch_threshold, eps_s=epsilon_start, eps_e=epsilon_end, sig_s=sigma_start, sig_e=sigma_end, ep_nb=epoch_nbr):
        self.threshold = threshold
        self.changed_connexions = 0
        self.alpha = alpha_start
        self.eta = eta_start
        self.alpha_stepping = (alpha_end - alpha_start) / ep_nb
        self.eta_stepping = (eta_end - eta_start) / ep_nb

        self.data = np.array(data)
        self.vector_list = None
        data_shape = self.data.shape[1]
        data_max = np.max(self.data)
        data_min = np.min(self.data)

        # Initializing the neural grid
        self.nodes = np.empty((neuron_nbr, neuron_nbr), dtype=DPCNeuron)
        for x in range(neuron_nbr):
            for y in range(neuron_nbr):
                self.nodes[x, y] = DPCNeuron(x, y, data_shape, data_min, data_max, connexion_matrices[x][y], available_neighbours(x, y))

        # Generating Connexions
        self.global_connections_graph = None
        self.neural_graph = None
        self.neural_adjacency_matrix = None
        self.neural_dist = None
        self.distance_vector = None
        self.refresh_distance_vector = True

        self.generate_global_connections_graph()
        self.neural_graph = self.global_connections_graph.extract_neurons_graph()
        self.compute_neurons_distance()

        if log_graphs:
            self.neural_graph.print_graph()
            print(self.neural_graph.to_string())
            print(self.neural_dist)

    def compute_neurons_distance(self):
        super().compute_neurons_distance()
        for i in range(len(self.neural_dist)):
            for j in range(len(self.neural_dist[0])):
                if self.neural_dist[i, j] == 1:
                    self.nodes[i % neuron_nbr, i//neuron_nbr].connected_list.append((j % neuron_nbr, j//neuron_nbr))

    def train(self, iteration, epoch_time, vector_coordinates, f=normalized_gaussian, distance=dist_quad):
        if iteration % epoch_time == 0 and iteration > 0:
            self.check_allegiance()

        vector = self.data[vector_coordinates]

        # Getting the Best matching unit
        bmu = self.winner(vector, distance)
        self.nodes[bmu].t = 1
        for i in range(len(self.distance_vector)):
            if np.sum(vector-self.nodes[bmu].weight) == 0:
                self.distance_vector[i] = 0
            else:
                self.distance_vector[i] = np.exp(-1/(elasticity**2)*(i/len(self.distance_vector))**2/(normalized_euclidean_norm(vector, self.nodes[bmu].weight))**2)
        if log_gaussian_vector:
            print(self.distance_vector)
        self.PC_updating_weights(bmu, vector)

        return bmu[0], bmu[1]

    def PC_updating_weights(self, bmu, vector, distance=dist_quad, f=normalized_gaussian):
        # updating BMU weights (using alpha instead of self.epsilon
        self.nodes[bmu].weight += self.alpha*(vector-self.nodes[bmu].weight)
        modified = np.zeros((neuron_nbr, neuron_nbr))
        modified[bmu] = 0
        current = [bmu,]
        d = 0
        while current:
            d += 1
            next = []
            for j in current:
                for i in self.nodes[j].connected_list:
                    if modified[i] != -1:
                        if modified[i] == 0:
                            next.append(i)
                        modified[i] += 1
            for j in current:
                for i in self.nodes[j].connected_list:
                    if modified[i] != -1:
                        diff = self.nodes[j].weight - self.nodes[i].weight
                        self.nodes[i].weight += self.alpha * diff * normalized_gaussian(self.distance_vector[d], self.eta*diff)/modified[i]
            for j in current:
                for i in self.nodes[j].connected_list:
                    modified[i] = -1
            current = next
