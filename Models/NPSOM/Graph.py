from Parameters import *
import numpy as np
import copy
from heapq import *
from functools import *
from scipy.sparse.csgraph import floyd_warshall


@total_ordering
class Edge:
    def __init__(self, in_vertex, out_vertex, weight):
        self.in_vertex = in_vertex
        self.out_vertex = out_vertex
        self.weight = weight

    def to_string(self):
        return self.in_vertex+"--"+str(self.weight)+"-->"+self.out_vertex

    def __eq__(self, other):
        return self.weight == other.weight

    def __lt__(self, other):
        return self.weight < other.weight

    def __le__(self, other):
        return self.weight <= other.weight

    def __gt__(self, other):
        return self.weight > other.weight

    def __ge__(self, other):
        return self.weight >= other.weight


class Graph:
    def __init__(self):
        self.edges_list = []
        self.vertex_list = {}

    def copy(self):
        return copy.deepcopy(self)

    def add_edge(self, edge):
        if edge.in_vertex not in self.vertex_list:
            self.vertex_list[edge.in_vertex] = len(self.vertex_list)
        if edge.out_vertex not in self.vertex_list:
            self.vertex_list[edge.out_vertex] = len(self.vertex_list)
        self.edges_list.append(edge)

    def add_vertex(self, vertex):
        if vertex not in self.vertex_list:
            self.vertex_list[vertex] = len(self.vertex_list)

    def remove_edge(self, vertex_in, vertex_out):
        for i in range(len(self.edges_list)):
            if self.edges_list[i].in_vertex == vertex_in and self.edges_list[i].out_vertex == vertex_out:
                self.edges_list.pop(i)
                break

    def get_adjacency_matrix(self):
        adjacency_matrix = np.full((len(self.vertex_list), len(self.vertex_list)), np.Infinity)
        for e in self.edges_list:
            if e.weight == 0:
                adjacency_matrix[self.vertex_list[e.in_vertex], self.vertex_list[e.out_vertex]] = 0.01
            else:
                adjacency_matrix[self.vertex_list[e.in_vertex], self.vertex_list[e.out_vertex]] = e.weight
        for v in self.vertex_list:
            adjacency_matrix[self.vertex_list[v], self.vertex_list[v]] = 0
        return adjacency_matrix

    def get_binary_adjacency_matrix(self):
        adjacency_matrix = np.full((len(self.vertex_list), len(self.vertex_list)), 0)
        for e in self.edges_list:
            adjacency_matrix[self.vertex_list[e.in_vertex], self.vertex_list[e.out_vertex]] = 1
        for v in self.vertex_list:
            adjacency_matrix[self.vertex_list[v], self.vertex_list[v]] = 0
        return adjacency_matrix

    def get_all_shortest_paths(self):
        return floyd_warshall(self.get_adjacency_matrix())

    def extract_neurons_graph(self):
        new_edges_list = []
        link_edges_list = []
        finished = []
        for e in self.edges_list:
            if e.in_vertex[0] == 'n':
                heappush(new_edges_list, e)
            else:
                link_edges_list.append(e)
        hash_map = [[] for i in range(len(self.vertex_list))]
        history = self.get_adjacency_matrix()
        for e in link_edges_list:
            hash_map[self.vertex_list[e.in_vertex]].append(e)
        while new_edges_list:
            current = heappop(new_edges_list)
            for e in hash_map[self.vertex_list[current.out_vertex]]:
                w = current.weight + e.weight
                if w < history[self.vertex_list[current.in_vertex], self.vertex_list[e.out_vertex]]:
                    history[self.vertex_list[current.in_vertex], self.vertex_list[e.out_vertex]] = w
                    new = Edge(current.in_vertex, e.out_vertex, w)
                    if new.out_vertex[0] == 'n':
                        finished.append(new)
                    else:
                        heappush(new_edges_list, new)
        g = Graph()
        for e in finished:
            g.add_edge(e)
        for y in range(neuron_nbr):
            for x in range(neuron_nbr):
                g.add_vertex("n"+str(x)+','+str(y))

        g.set_neuron_id()
        return g

    def set_neuron_id(self):
        for v in self.vertex_list.keys():
            str = v[1:].split(',')
            self.vertex_list[v] = int(str[1])*neuron_nbr+int(str[0])

    def to_string(self):
        res = "Vertices : "+str(self.vertex_list)+"\nEdges :\n"
        for e in self.edges_list:
            res += e.to_string()+"\n"
        return res

    def print_graph(self):
        res = ""
        adj = self.get_adjacency_matrix()
        for i in range(len(adj)):
            for j in range(len(adj[0])):
                if adj[i][j] == np.Infinity:
                    res += "0 "
                else:
                    res += str(int(adj[i][j]))+" "
            res += "\n"
        print(res)
