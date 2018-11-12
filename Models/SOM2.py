from Models.Graph import *
import copy

def dist_quad(x, y):
    assert np.array_equal(np.array(x.shape), np.array(y.shape))
    return np.sum((x - y) ** 2)

def manhattan_dist(x, y):
    return np.sum(np.abs(x - y))

def gauss(d, sig):
    return np.exp(-((d / sig) ** 2) / 2) / sig

def normalized_gaussian(d, sig):
    return np.exp(-((d / sig) ** 2) / 2)

class Neurone:
    def __init__(self, x, y, shape, min, max, connections):
        self.x = x  # Positions in the grid
        self.y = y
        self.t = 1  # Time elapsed since last selected as BMU
        self.connection_matrix = connections
        self.weight = (max-min) * np.random.random(shape) + min

class SOM2:
    def __init__(self, data, connexion_matrices, psom=False, pcsom=False, eps_s=epsilon_start, eps_e=epsilon_end, sig_s=sigma_start, sig_e=sigma_end, alpha_s=alpha_start, alpha_e=alpha_end, eta_s=eta_start, eta_e=eta_end, ep_nb=epoch_nbr):
        self.nbr_removed=0
        self.epsilon = eps_s
        self.epsilon_stepping = (eps_e - eps_s) / ep_nb
        if pcsom_decreasing_param:
            self.alpha = alpha
            self.eta = eta
        else:
            self.alpha = alpha_s
            self.eta = eta_s
        self.alpha_stepping = (alpha_e - alpha_s) / ep_nb
        self.sigma = sig_s
        self.sigma_stepping = (sig_e - sig_s) / ep_nb
        self.eta_stepping = (eta_e - eta_s) / ep_nb
        self.data = np.array(data)
        self.vector_list = None
        self.psom = psom
        self.pcsom = pcsom
        data_shape = self.data.shape[1]
        data_max = np.max(self.data)
        data_min = np.min(self.data)
        # Initializing the neural grid
        self.nodes = np.empty((neuron_nbr, neuron_nbr), dtype=Neurone)
        for x in range(neuron_nbr):
            for y in range(neuron_nbr):
                self.nodes[x, y] = Neurone(x, y, data_shape, data_min, data_max, connexion_matrices[x,y])

        # Generating Connexions
        self.global_connections_graph = None
        self.neural_graph = None
        self.neural_adjacency_matrix = None
        self.neural_dist = None
        self.distance_vector = None

        self.generate_global_connections_graph()
        self.neural_graph = self.global_connections_graph.extract_neurons_graph()
        self.compute_neurons_distance()

        if log_graphs:
            self.neural_graph.print_graph()
            print(self.neural_graph.to_string())
            print(self.neural_dist)

    def copy(self):
        return copy.deepcopy(self)

    def generate_global_connections_graph(self):
        self.global_connections_graph = Graph()
        for x in range(neuron_nbr):  # Creating the links between inputs and outputs
            for y in range(neuron_nbr):
                if y != 0:
                    self.global_connections_graph.add_edge(Edge("No"+str(x)+","+str(y), "Si"+str(x)+","+str(y-1), 0))
                if y != neuron_nbr-1:
                    self.global_connections_graph.add_edge(Edge("So"+str(x)+","+str(y), "Ni"+str(x)+","+str(y+1), 0))
                if x != neuron_nbr-1:
                    self.global_connections_graph.add_edge(Edge("Eo"+str(x)+","+str(y), "Wi"+str(x+1)+","+str(y), 0))
                if x != 0:
                    self.global_connections_graph.add_edge(Edge("Wo"+str(x)+","+str(y), "Ei"+str(x-1)+","+str(y), 0))

        for x in range(neuron_nbr):
            for y in range(neuron_nbr):
                for i in range(5):
                    for j in range(5):
                        if self.nodes[x, y].connection_matrix[i, j] != 0:
                            input_vertex = SOM.get_index(i, 'i')+str(x)+','+str(y)
                            output_vertex = SOM.get_index(j, 'o')+str(x)+','+str(y)
                            e = Edge(input_vertex, output_vertex, SOM.neurons_only_weight(i, j))
                            self.global_connections_graph.add_edge(e)

    def compute_neurons_distance(self):
        self.neural_adjacency_matrix = self.neural_graph.get_adjacency_matrix()
        self.neural_dist = self.neural_graph.get_all_shortest_paths()
        self.neural_dist = self.neural_dist.astype(int)  # /!\there is a numpy bug that converts inf to a negative value
        self.distance_vector = np.empty(np.max(self.neural_dist)+1, dtype=float)

    @staticmethod
    def uniform_weight(i, j):
        return 1

    @staticmethod
    def neurons_only_weight(i, j):
        if i == 4 or j == 4:
            return 0.5
        return 0

    @staticmethod
    def get_index(index, type):
        return {
            0: "N"+type,    # North
            1: "E"+type,    # East
            2: "W"+type,    # West
            3: "S"+type,    # South
            4: "n"          # neuron
        }.get(index)

    def winner(self, vector, distance=dist_quad):
        dist = np.empty_like(self.nodes, dtype=float)
        for x in range(neuron_nbr):  # Computes the distances between the tested vector and all nodes
            for y in range(neuron_nbr):
                self.nodes[x, y].t += 1
                dist[x, y] = distance(self.nodes[x, y].weight, vector)
        return np.unravel_index(np.argmin(dist, axis=None), dist.shape)  # Returning the Best Matching Unit's index.

    def winners(self):
        datacomp = np.zeros(len(self.data), dtype=int)  # datacomp est la liste du numero du neurone vainqueur pour l'imagette correspondante
        for i in range(len(self.data)):
            bmu = self.winner(self.data[i])
            datacomp[i] = bmu[1]*neuron_nbr+bmu[0]
        return datacomp

    def train(self, iteration, epoch_time, vector_coordinates, f=normalized_gaussian, distance=dist_quad):
        if (iteration % epoch_time == 0):
            if (iteration>0):
                self.epsilon += self.epsilon_stepping
                self.sigma += self.sigma_stepping
                if pcsom_decreasing_param:
                    self.alpha += self.alpha_stepping
                    self.eta += self.eta_stepping
                if (self.psom):
                    print("PSOM pruning")
                    self.pruning_neighbors()
                if (self.pcsom):
                    print("PCSOM pruning")
                    self.pruning_neighbors()
            for i in range(len(self.distance_vector)):
                self.distance_vector[i] = f(i/(len(self.distance_vector)-1), self.sigma)
            if log_gaussian_vector:
                print(self.distance_vector)

        vector = self.data[vector_coordinates]

        # Getting the Best matching unit
        bmu = self.winner(vector, distance)
        self.nodes[bmu].t = 1
        if (self.pcsom):
            self.PC_updating_weights(bmu, vector,distance)
        else:
            self.updating_weights(bmu, vector)
        return bmu[0], bmu[1]

    def PC_updating_weights(self, bmu, vector,distance=dist_quad, f=normalized_gaussian):
        # filling a tab with distances from BMU
        maxdist=0
        dists = np.empty((neuron_nbr,neuron_nbr), dtype=np.ndarray)
        for i in range(neuron_nbr): 
            for j in range(neuron_nbr):
                dists[i,j] = self.neural_dist[bmu[1]*neuron_nbr+bmu[0], j*neuron_nbr+i]
                if (dists[i,j]>maxdist):
                    maxdist = dists[i,j]
        # updating BMU weights (using alpha instead of self.epsilon
        self.nodes[bmu[0],bmu[1]].weight += self.alpha*(vector-self.nodes[bmu[0],bmu[1]].weight)
        # attention d'apres papier AHS : il faudrait multiplier encore par distance(vector,bmu_weight)
        # Updating weights of all nodes in cellular mode
        data_shape=self.data.shape[1]
        for d in range(1,maxdist):
            for i in range(neuron_nbr):
                for j in range(neuron_nbr):
                    if (dists[i,j]==d):
                        #look for influential neurons
                        nbr_inf=0
                        update=np.zeros(data_shape,dtype=float)
                        #North
                        if (i>0):
                            if (dists[i-1,j]==d-1):
                                nbr_inf += 1
                                update += (self.nodes[i-1,j].weight-self.nodes[i,j].weight)*np.exp(-d/(self.eta*distance(self.nodes[i-1,j].weight,self.nodes[i,j].weight)))
                        #East
                        if (j<neuron_nbr-1):
                            if (dists[i,j+1]==d-1):
                                nbr_inf += 1
                                update += (self.nodes[i,j+1].weight-self.nodes[i,j].weight)*np.exp(-d/(self.eta*distance(self.nodes[i,j+1].weight,self.nodes[i,j].weight)))
                        #West
                        if (j>0):
                            if (dists[i,j-1]==d-1):
                                nbr_inf += 1
                                update += (self.nodes[i,j-1].weight-self.nodes[i,j].weight)*np.exp(-d/(self.eta*distance(self.nodes[i,j-1].weight,self.nodes[i,j].weight)))
                        #South
                        if (i<neuron_nbr-1):
                            if (dists[i+1,j]==d-1):
                                nbr_inf += 1
                                update += (self.nodes[i+1,j].weight-self.nodes[i,j].weight)*np.exp(-d/(self.eta*distance(self.nodes[i+1,j].weight,self.nodes[i,j].weight)))
                        if (nbr_inf==0):
                            print("ARGL : no influencial neuron\n")
                        self.nodes[i, j].weight += self.alpha*update/nbr_inf

    def updating_weights(self, bmu, vector):
        for x in range(neuron_nbr):  # Updating weights of all nodes
            for y in range(neuron_nbr):
                dist = self.neural_dist[bmu[1]*neuron_nbr+bmu[0], y*neuron_nbr+x]
                if dist >= 0:  # exploiting here the numpy bug so that negative value equals no connections
                    self.nodes[x, y].weight += self.epsilon*self.distance_vector[dist]*(vector-self.nodes[x, y].weight)

    def pruning_neighbors(self):
        for x in range(neuron_nbr-1):
            for y in range(neuron_nbr-1):
                self.pruning_check(x, y, x+1, y)
                self.pruning_check(x, y, x, y+1)
        for x in range(neuron_nbr-1):
            self.pruning_check(x, neuron_nbr-1, x+1, neuron_nbr-1)
        for y in range(neuron_nbr-1):
            self.pruning_check(neuron_nbr-1, y, neuron_nbr-1, y+1)
        self.compute_neurons_distance()

    def pruning_check(self, x1, y1, x2, y2):
        one = y1*neuron_nbr + x1
        two = y2*neuron_nbr + x2
        if (self.psom):
            omega=psom_omega
        if (self.pcsom):
            omega=pcsom_omega
        if self.neural_adjacency_matrix[one, two] != 0 and self.neural_adjacency_matrix[one, two] != np.inf:
            diff = manhattan_dist(self.nodes[x1, y1].weight, self.nodes[x2, y2].weight)
            probability = np.exp(-1/omega * 1/(diff * self.nodes[x1, y1].t * self.nodes[x2, y2].t))
            if np.random.rand() < probability:
                print("Removed (", x1, ",", y1, ") - (", x2, ",", y2, ") probability : ", probability)
                self.remove_edges((x1, y1), (x2, y2))
                self.nbr_removed += 1

    def remove_edges(self, v1, v2):  # remove_edges((x, y), (x2, y2))
        inp = "n"+str(v1[0])+','+str(v1[1])
        out = "n"+str(v2[0])+','+str(v2[1])
        self.neural_graph.remove_edge(inp, out)
        self.neural_graph.remove_edge(out, inp)

    def fully_random_vector(self):
        return np.random.randint(np.shape(self.data)[0])

    def unique_random_vector(self):
        return self.vector_list.pop(0)

    def generate_random_list(self):
        self.vector_list = list(range(len(self.data)))
        np.random.shuffle(self.vector_list)
    
    def get_som_as_map(self):
        result = np.empty((neuron_nbr, neuron_nbr), dtype=np.ndarray)
        for x in range(neuron_nbr):
            for y in range(neuron_nbr):
                result[x, y] = self.nodes[x, y].weight
        return result
    
    def get_som_as_list(self):
        result = np.empty(neuron_nbr*neuron_nbr, dtype=np.ndarray)
        for x in range(neuron_nbr):
            for y in range(neuron_nbr):
                result[y*neuron_nbr + x] = self.nodes[x, y].weight
        return result

    def set_som_as_list(self, list):
        for x in range(neuron_nbr):
            for y in range(neuron_nbr):
                self.nodes[x, y].weight = list[y*neuron_nbr + x]
