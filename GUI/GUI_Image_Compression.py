from GUI.SOM_View import *

import time

width = 600
height = 600


class GraphicalSOM:
    def __init__(self):
        self.window = Tk()
        self.window.title("Graphical Self-Organised Map")
        self.canvas = Canvas(self.window, width=width, height=height, bg="ivory")
        self.canvas.grid(row=0, column=0, columnspan=10, rowspan=10, padx=10, pady=10)

        np.random.seed(0)
        self.img = Dataset("./image/limited_test/peppers.pgm")
        data = self.img.data

        self.SOM = SOM(data, kohonen())

        self.epoch_time = len(self.SOM.data)
        self.current_iteration = 0
        self.total_iterations = self.epoch_time * epoch_nbr
        self.running = False

        self.draw_result()
        self.draw_buttons()
        self.draw_metrics()
        self.mapView = SOMView(self.SOM)


        self.canvas.update()
        self.window.mainloop()

    def draw_result(self):
        self.im = self.img.compute_result(self.SOM)
        self.im = self.im.resize((width, height))
        self.ph = itk.PhotoImage(self.im)
        self.canvas.create_image(0, 0, image=self.ph, anchor=NW)

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
        self.dataset_str.set("Dataset size : "+str(len(self.SOM.data)))
        winners = self.SOM.winners()
        self.mean_error_str.set("Mean error : "+str(self.SOM.compute_mean_error(winners)))
        self.psnr_str.set("PSNR : "+str(self.SOM.peak_signal_to_noise_ratio(winners)))

    def refresh(self):
        self.canvas.delete("all")
        self.draw_result()
        self.canvas.update()
        # self.update_metrics()
        self.mapView.refresh()

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
                self.refresh()
                start_time = time.time()


GraphicalSOM()
