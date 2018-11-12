from Models.PruningSOM import *

# PruningSOM Variant with neurons neurons going closer to their neighbours weights at each learning iteration instead
# of the training vector.
# Should learn really fast (like 1 or 2 epoch)


class PCSOM(PruningSOM):
    def __init__(self, data, connexion_matrices, eps_s=epsilon_start, eps_e=epsilon_end, sig_s=sigma_start, sig_e=sigma_end, alpha_s=alpha_start, alpha_e=alpha_end, eta_s=eta_start, eta_e=eta_end, ep_nb=epoch_nbr):
        self.nbr_removed = 0
        if pcsom_decreasing_param:
            self.alpha = alpha_s
            self.eta = eta_s
            self.alpha_stepping = (alpha_e - alpha_s) / ep_nb
            self.eta_stepping = (eta_e - eta_s) / ep_nb
        else:
            self.alpha = alpha
            self.eta = eta
        super().__init__(data, connexion_matrices, eps_s, eps_e, sig_s, sig_e, ep_nb)

    def train(self, iteration, epoch_time, vector_coordinates, f=normalized_gaussian, distance=dist_quad):
        if iteration % epoch_time == 0:
            if iteration > 0:
                self.epsilon += self.epsilon_stepping
                self.sigma += self.sigma_stepping
                if pcsom_decreasing_param:
                    self.alpha += self.alpha_stepping
                    self.eta += self.eta_stepping
                self.pruning_neighbors()
            for i in range(len(self.distance_vector)):
                self.distance_vector[i] = f(i/(len(self.distance_vector)-1), self.sigma)
            if log_gaussian_vector:
                print(self.distance_vector)

        vector = self.data[vector_coordinates]

        # Getting the Best matching unit
        bmu = self.winner(vector, distance)
        self.nodes[bmu].t = 1
        self.PC_updating_weights(bmu, vector,distance)
        return bmu[0], bmu[1]

    def updating_weights(self, bmu, vector):
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