from Models.SOM import *


class Local_SOM(SOM):
    updates_list = []
    current_vector = [0, 0]

    def winner(self, vector, distance=dist_quad):
        self.updates_list = []
        self.current_vector = vector
        cont = True
        bmu = (np.random.randint(0,neuron_nbr), np.random.randint(0,neuron_nbr))
        best = distance(self.nodes[bmu].weight, vector)
        while cont:
            self.updates_list.append(bmu)
            cont = False
            dists = self.neural_dist[bmu[0] * neuron_nbr + bmu[1]]
            print(dists)
            for i in range(neuron_nbr*neuron_nbr):
                if dists[i] == 1:
                    ij = np.unravel_index(i, self.nodes.shape)
                    value = distance(self.nodes[ij].weight, vector)
                    if value < best:
                        cont = True
                        best = value
                        bmu = ij
        return bmu
