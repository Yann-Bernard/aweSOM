from Models.Connections_Models import *
from Models.Parameters import *


def kohonen():
    connexion_matrix = np.empty((neuron_nbr, neuron_nbr, 5, 5))
    for y in range(neuron_nbr):
        for x in range(neuron_nbr):
            connexion_matrix[x, y] = kohonen_matrix
    return connexion_matrix


def small_worlds():
    connexion_matrix = np.empty((neuron_nbr, neuron_nbr, 5, 5))
    pattern = [[top_left, top_highway, top_right],
               [left_highway, kohonen_matrix, right_highway],
               [bottom_left, bottom_highway, bottom_right]]
    for y in range(neuron_nbr):
        for x in range(neuron_nbr):
            connexion_matrix[x, y] = pattern[y % 3][x % 3]
    return connexion_matrix


def acentric_small_worlds():
    connexion_matrix = np.empty((neuron_nbr, neuron_nbr, 5, 5))
    pattern = [[top_left, kohonen_matrix, top_right],
               [kohonen_matrix, kohonen_matrix, kohonen_matrix],
               [bottom_left, kohonen_matrix, bottom_right]]
    for y in range(neuron_nbr):
        for x in range(neuron_nbr):
            connexion_matrix[x, y] = pattern[y % 3][x % 3]
    return connexion_matrix


def fully_connected_small_worlds():
    connexion_matrix = np.empty((neuron_nbr, neuron_nbr, 5, 5))
    pattern = [[fc_top_left, fc_top, fc_top_right],
               [fc_left, fc, fc_right],
               [fc_bottom_left, fc_bottom, fc_bottom_right]]
    for y in range(neuron_nbr):
        for x in range(neuron_nbr):
            connexion_matrix[x, y] = pattern[y % 3][x % 3]
    return connexion_matrix


def star():
    connexion_matrix = np.empty((neuron_nbr, neuron_nbr, 5, 5))
    pattern = [[star_corner_left, star_top, star_corner_right],
               [star_left, kohonen_matrix, star_right],
               [star_corner_left, star_bottom, star_corner_right]]
    for y in range(neuron_nbr):
        for x in range(neuron_nbr):
            connexion_matrix[x, y] = pattern[y % 3][x % 3]
    return connexion_matrix


def random():
    connexion_matrix = np.empty((neuron_nbr, neuron_nbr, 5, 5))
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
            connexion_matrix[x, y] = matrix
    return connexion_matrix