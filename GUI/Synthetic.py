from tkinter import *
from tkinter.ttk import *
from Data.Generated_2D import *
from Models.NPSOM.Connections import *
from Models.DynamicPCSOM import *
from Models.PCSOM import *

import colorsys
import time

radius = 2
width = 600
height = 600
connexions_size = 300
dataset_size = 2000


def draw_data_point(canvas, x, y, fill_color, tag):
    canvas.create_oval(x-radius, y-radius, x+radius, y+radius, outline=fill_color, fill=fill_color, tags=tag)


def project(x, y):
    return radius+x*(width-radius), radius+y*(height-radius)


def convert_color(srgb):
    return "#"+'%02x' % int(srgb[0]*255) + '%02x' % int(srgb[1]*255) + '%02x' % int(srgb[2]*255)


class GraphicalSOM:
    def __init__(self):
        self.window = Tk()
        self.window.title("Graphical Self-Organised Map")
        self.canvas = Canvas(self.window, width=width, height=height, bg="ivory")
        self.canvas.grid(row=0, column=0, columnspan=10, rowspan=10, padx=10, pady=10)
        self.connexion = Canvas(self.window, width=connexions_size, height=connexions_size, bg="ivory")
        self.connexion.grid(row=6, column=10, columnspan=1, rowspan=1, padx=10, pady=10)
        np.random.seed(0)
        self.SOM = DynamicSOM(sierpinski_carpet(dataset_size, 1), star())
        self.epoch_time = len(self.SOM.data)
        self.current_iteration = 0
        self.total_iterations = self.epoch_time * epoch_nbr
        self.running = False
        self.colors = [None] * (neuron_nbr//3)**2
        self.outline = [None] * (neuron_nbr//3)**2
        for i in range(len(self.colors)):
            self.colors[i] = convert_color(colorsys.hsv_to_rgb(i/len(self.colors), 1.0, 0.5))
            self.outline[i] = convert_color(colorsys.hsv_to_rgb(i/len(self.colors), 1.0, 1.0))

        self.draw_buttons()
        self.draw_data()
        self.draw_map()
        self.draw_metrics()
        self.draw_connexions()

        self.item = None
        self.canvas.bind("<Button-1>", self.deselect)
        self.canvas.bind("<Button-3>", self.selection)
        self.canvas.bind("<Button-2>", self.move_neuron)
        self.window.bind('c', self.create_edge)
        self.window.bind('d', self.delete_edge)

        self.canvas.update()
        self.connexion.update()
        self.window.mainloop()

    def draw_buttons(self):
        self.number = Entry(self.window, text="1")
        self.listbox = Combobox(self.window, state="readonly")
        self.listbox['values'] = ("iterations", "epochs")
        self.listbox.current(0)
        self.step = Button(self.window, text='Step-by-step', command=self.run_once)
        self.run = Button(self.window, text='Run', command=self.run)

        self.number.grid(row=10, column=0)
        self.listbox.grid(row=10, column=1)
        self.step.grid(row=10, column=2)
        self.run.grid(row=10, column=3)

    def draw_data(self):
        data = self.SOM.data
        for element in data:
            x, y = project(element[0], element[1])
            draw_data_point(self.canvas, x, y, "red", "data")

    def draw_map(self):
        positions = np.empty((neuron_nbr, neuron_nbr), dtype=list)
        adjacency_matrix = self.SOM.neural_adjacency_matrix
        for y in range(neuron_nbr):
            for x in range(neuron_nbr):
                positions[x, y] = project(self.SOM.nodes[x, y].weight[0], self.SOM.nodes[x, y].weight[1])
                draw_data_point(self.canvas, positions[x, y][0], positions[x, y][1], "blue", ("neuron", str(x)+";"+str(y)))
                for k in range(y*neuron_nbr+x):
                    if adjacency_matrix[y*neuron_nbr+x][k] == 1:
                        x1 = k % neuron_nbr
                        y1 = k//neuron_nbr
                        self.canvas.create_line(positions[x, y][0], positions[x, y][1], positions[x1, y1][0], positions[x1, y1][1],
                                                fill="blue", tags=("link", str(x)+";"+str(y)+";"+str(x1)+";"+str(y1)))

    def draw_connexions(self):
        self.connexion.delete("all")
        size = connexions_size/neuron_nbr
        for y in range(neuron_nbr):
            for x in range(neuron_nbr):
                if x % 3 == 1 and y % 3 == 1:
                    self.connexion.create_rectangle(x*size, y*size, (x+1)*size, (y+1)*size, outline=self.outline[y//3 * neuron_nbr//3 + x//3], fill=self.colors[y//3 * neuron_nbr//3 + x//3])
                else:
                    center = self.SOM.nodes[x, y].neighbour[self.SOM.nodes[x, y].current_center]
                    self.connexion.create_rectangle(x*size, y*size, (x+1)*size, (y+1)*size, outline=self.outline[center[1]//3 * neuron_nbr//3 + center[0]//3], fill=self.colors[center[1]//3 * neuron_nbr//3 + center[0]//3])

    def draw_metrics(self):
        self.ite_str = StringVar()
        self.epoch_str = StringVar()
        self.neurons_str = StringVar()
        self.dataset_str = StringVar()
        self.mean_error_str = StringVar()
        self.psnr_str = StringVar()

        self.ite_label = Label(textvariable=self.ite_str)
        self.epoch_label = Label(textvariable=self.epoch_str)
        self.neurons_label = Label(textvariable=self.neurons_str)
        self.dataset_label = Label(textvariable=self.dataset_str)
        self.mean_error_label = Label(textvariable=self.mean_error_str)
        self.psnr_label = Label(textvariable=self.psnr_str)

        self.ite_label.grid(column=10, row=0)
        self.epoch_label.grid(column=10, row=1)
        self.neurons_label.grid(column=10, row=2)
        self.dataset_label.grid(column=10, row=3)
        self.mean_error_label.grid(column=10, row=4)
        self.psnr_label.grid(column=10, row=5)
        self.update_metrics()

    def update_metrics(self):
        self.ite_str.set("Iteration number : "+str(self.current_iteration))
        self.epoch_str.set("Epochs number : "+str(self.current_iteration//self.epoch_time))
        self.neurons_str.set("Neurons number : "+str(neuron_nbr*neuron_nbr))
        self.dataset_str.set("Dataset size : "+str(dataset_size))
        winners = self.SOM.get_all_winners()
        self.mean_error_str.set("Mean error : "+str(self.SOM.compute_mean_error(winners)))
        self.psnr_str.set("PSNR : "+str(self.SOM.peak_signal_to_noise_ratio(winners)))

    def deselect(self, event):
        if self.item and "link" in self.canvas.gettags(self.item):
            self.canvas.itemconfig(self.item, fill="blue")
        elif self.item and "neuron" in self.canvas.gettags(self.item):
            self.canvas.itemconfig(self.item, fill="blue", outline="blue")
        self.item = None

    def selection(self, event):
        self.deselect(event)
        self.item = self.canvas.find_closest(event.x, event.y)
        if self.item and "link" in self.canvas.gettags(self.item):
            self.canvas.itemconfig(self.item, fill="green")
        elif self.item and "neuron" in self.canvas.gettags(self.item):
            self.canvas.itemconfig(self.item, fill="green", outline="green")

    def delete_edge(self, event):
        if self.item and "link" in self.canvas.gettags(self.item):
            print(self.canvas.gettags(self.item))
            coords = self.canvas.gettags(self.item)[1]
            coords = coords.split(';')
            self.SOM.remove_edges((int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3])))
            self.SOM.compute_neurons_distance()
            self.canvas.delete(self.item)
            self.item = None
            self.canvas.update()

    def create_edge(self, event):
        print(event.x-10, event.y-10)
        target = self.canvas.find_closest(event.x-10, event.y-10)
        if self.item and target and "neuron" in self.canvas.gettags(self.item) and "neuron" in self.canvas.gettags(target):
            print(self.canvas.gettags(self.item), self.canvas.gettags(target))

            src_coords = self.canvas.gettags(self.item)[1]
            src_coords = src_coords.split(';')
            tgt_coords = self.canvas.gettags(target)[1]
            tgt_coords = tgt_coords.split(';')

            self.SOM.create_edges((int(src_coords[0]), int(src_coords[1])), (int(tgt_coords[0]), int(tgt_coords[1])))
            self.SOM.compute_neurons_distance()
            self.refresh()

    def move_neuron(self, event):
        if self.item and "neuron" in self.canvas.gettags(self.item):
            print(self.canvas.gettags(self.item))
            coords = self.canvas.gettags(self.item)[1]
            coords = coords.split(';')
            self.SOM.nodes[int(coords[0]), int(coords[1])].weight = np.array([event.x/width, event.y/height])
            self.refresh()

    def refresh(self):
        self.canvas.delete("all")
        self.draw_data()
        self.draw_map()
        #self.draw_connexions()
        self.canvas.update()
        self.connexion.update()
        self.update_metrics()

    def run_once(self):
        try:
            iterations = int(self.number.get())
        except ValueError:
            iterations = 0
        if self.listbox.get() == "epochs":
            iterations = iterations * self.epoch_time
        if iterations + self.current_iteration > self.total_iterations:
            iterations = self.total_iterations - self.current_iteration
        for i in range(iterations):
            if self.current_iteration % self.epoch_time == 0:
                self.SOM.generate_random_list()
            vect = self.SOM.unique_random_vector()
            self.SOM.train(self.current_iteration, self.epoch_time, vect)
            self.current_iteration += 1
        self.refresh()

    def run(self):
        start_time = time.time()
        while self.current_iteration < self.total_iterations:
            if self.current_iteration % self.epoch_time == 0:
                self.SOM.generate_random_list()
                self.update_metrics()
            vect = self.SOM.unique_random_vector()
            self.SOM.train(self.current_iteration, self.epoch_time, vect)
            self.current_iteration += 1
            if time.time() - start_time > 0.3:
                self.canvas.delete("all")
                self.draw_data()
                self.draw_map()
                #self.draw_connexions()
                self.canvas.update()
                self.connexion.update()
                start_time = time.time()

