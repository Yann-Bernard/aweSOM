from Models.SOM import *

# Dynamic SOM version with elasticity for the connections (based on Nicolas Rougier's paper)
# combined with center switching of the Generalized Star architecture.


def available_neighbours(x, y):
    res = {}
    if x % 3 == 1 and y % 3 == 1:
        return res
    res['C'] = ((x // 3) * 3 + 1, (y // 3) * 3 + 1)
    if y % 3 <= 1 and y >= 3:
        res['N'] = ((x//3)*3+1, (y//3-1)*3+1)
    if y % 3 >= 1 and y <= neuron_nbr-3:
        res['S'] = ((x//3)*3+1, (y//3+1)*3+1)
    if x % 3 <= 1 and x >= 3:
        res['W'] = ((x//3-1)*3+1, (y//3)*3+1)
    if x % 3 >= 1 and x <= neuron_nbr-3:
        res['E'] = ((x//3+1)*3+1, (y//3)*3+1)
    return res


class DynamicNeuron:
    def __init__(self, x, y, shape, min, max, connections, neighbour):
        self.x = x  # Positions in the grid
        self.y = y
        self.t = 1  # Time elapsed since last selected as BMU
        self.connection_matrix = connections
        self.weight = (max-min) * np.random.random(shape) + min
        self.current_center = 'C'
        self.neighbour = neighbour
        self.error = 0
        self.nb_BMU = 0


class DynamicSOM(SOM):
    def __init__(self, data, connexion_matrices, threshold=switch_threshold, eps_s=epsilon_start, eps_e=epsilon_end, sig_s=sigma_start, sig_e=sigma_end, ep_nb=epoch_nbr):
        self.threshold = threshold
        self.changed_connexions = 0

        self.data = np.array(data)
        self.vector_list = None
        data_shape = self.data.shape[1]
        data_max = np.max(self.data)
        data_min = np.min(self.data)

        # Initializing the neural grid
        self.nodes = np.empty((neuron_nbr, neuron_nbr), dtype=DynamicNeuron)
        for x in range(neuron_nbr):
            for y in range(neuron_nbr):
                self.nodes[x, y] = DynamicNeuron(x, y, data_shape, data_min, data_max, connexion_matrices[x][y], available_neighbours(x, y))

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

    def check_allegiance(self):
        for x in range(neuron_nbr):
            for y in range(neuron_nbr):
                if self.nodes[x, y].neighbour != {}:
                    min_distance = np.inf
                    min_index = ()
                    min_key = ''
                    for key, value in self.nodes[x, y].neighbour.items():
                        d = dist_quad(self.nodes[x, y].weight, self.nodes[value[0], value[1]].weight)
                        if d < min_distance:
                            min_distance = d
                            min_index = value
                            min_key = key
                    center_index = self.nodes[x, y].neighbour[self.nodes[x, y].current_center]
                    if min_distance * self.threshold < dist_quad(self.nodes[x, y].weight, self.nodes[center_index[0], center_index[1]].weight):
                        # print("Removed (", x, ",", y, ") - (", center_index[0], ",", center_index[1], ")")
                        # print("Created (", x, ",", y, ") - (", min_index[0], ",", min_index[1], ")")
                        self.changed_connexions += 1
                        self.remove_edges((x, y), center_index)
                        self.create_edges((x, y), min_index)
                        self.nodes[x, y].current_center = min_key
        self.compute_neurons_distance()

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
        self.updating_weights(bmu, vector)

        return bmu[0], bmu[1]

    def updating_weights(self, bmu, vector):
        for x in range(neuron_nbr):  # Updating weights of all nodes
            for y in range(neuron_nbr):
                dist = self.neural_dist[bmu[1]*neuron_nbr+bmu[0], y*neuron_nbr+x]
                if dist >= 0:  # exploiting here the numpy bug so that negative value equals no connections
                    self.nodes[x, y].weight += dsom_epsilon*normalized_euclidean_norm(vector, self.nodes[x,y].weight)*self.distance_vector[dist]*(vector-self.nodes[x, y].weight)

    def print_connexions(self):
        for x in range(neuron_nbr):
            for y in range(neuron_nbr):
                print(self.nodes[x, y].current_center, end='')
            print('')
