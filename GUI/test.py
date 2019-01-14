from Models.SOM import *
from tkinter import *
from Data.Simple_Data_Sample import *
from Models.NPSOM.Connections import *

h_l = 700
w_l = 700

h_r = 350
w_r = 350

def cercle(can, x, y, r, remplissage):
    #tracer un cercle de centre (x,y) et de rayon r
    can.create_oval(x-r, y-r, x+r, y+r, outline=remplissage,fill=remplissage)

    
def draw_SOM(n,map,can,h,w):
    
    mat = map.global_connections.extract_neurons_graph()
    
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            
            x_n = w//(n+1)+i*w//(n+1)
            y_n = h//(n+1)+j*h//(n+1)
            cercle(can, x_n, y_n, h//100, 'red')
            
            x_l = x_n - w//200
            x_r = x_n + w//200
            y_a = y_n - h//55
            y_b = y_n + h//55
            cercle(can, x_l, y_a, h//300, 'black')
            cercle(can, x_r, y_a, h//300, 'black')
            cercle(can, x_l, y_b, h//300, 'black')
            cercle(can, x_r, y_b, h//300, 'black')
            
            y_l = y_n - h//200
            y_r = y_n + h//200
            x_a = x_n - w//55
            x_b = x_n + w//55
            cercle(can, x_a, y_l, h//300, 'black')
            cercle(can, x_a, y_r, h//300, 'black')
            cercle(can, x_b, y_l, h//300, 'black')
            cercle(can, x_b, y_r, h//300, 'black')
            
            #TODO: create lines from local connections matrix

def draw_map(n,data,map,can,h,w):
    
    for i in range (len(data)):
            
        x_d = data[i][0]*(w-100)+(w-100)//10
        y_d = data[i][1]*(h-100)+(h-100)//10
        cercle(can, x_d, y_d, h//100, 'red')
    
    for i in range(neuron_nbr-1):
        for j in range(neuron_nbr-1):
            #print(map.nodes[i][j].weight)
            x_ij = map.nodes[i][j].weight[0]*(w-100)+(w-100)//10
            y_ij = map.nodes[i][j].weight[1]*(h-100)+(h-100)//10
            x_i1j = map.nodes[i+1][j].weight[0]*(w-100)+(w-100)//10
            y_i1j = map.nodes[i+1][j].weight[1]*(h-100)+(h-100)//10
            
            x_ij1 = map.nodes[i][j+1].weight[0]*(w-100)+(w-100)//10
            y_ij1 = map.nodes[i][j+1].weight[1]*(h-100)+(h-100)//10
            
            x_i1j1 = map.nodes[i+1][j+1].weight[0]*(w-100)+(w-100)//10
            y_i1j1 = map.nodes[i+1][j+1].weight[1]*(h-100)+(h-100)//10

            cercle(can, x_ij, y_ij, h//100, 'blue')
            cercle(can, x_i1j, y_i1j, h//100, 'blue')
            cercle(can, x_ij1, y_ij1, h//100, 'blue')
            cercle(can, x_i1j1, y_i1j1, h//100, 'blue')

            can.create_line(x_ij,y_ij,x_i1j,y_i1j,fill="blue")
            can.create_line(x_ij,y_ij,x_ij1,y_ij1,fill="blue")

            x_4j = map.nodes[neuron_nbr-1][j].weight[0]*(w-100)+(w-100)//10
            y_4j = map.nodes[neuron_nbr-1][j].weight[1]*(h-100)+(h-100)//10
            x_4j1 = map.nodes[neuron_nbr-1][j+1].weight[0]*(w-100)+(w-100)//10
            y_4j1 = map.nodes[neuron_nbr-1][j+1].weight[1]*(h-100)+(h-100)//10
            can.create_line(x_4j,y_4j,x_4j1, y_4j1,fill="blue")
            
            x_i4 = map.nodes[i][neuron_nbr-1].weight[0]*(w-100)+(w-100)//10
            y_i4 = map.nodes[i][neuron_nbr-1].weight[1]*(h-100)+(h-100)//10
            x_i14 = map.nodes[i+1][neuron_nbr-1].weight[0]*(w-100)+(w-100)//10
            y_i14 = map.nodes[i+1][neuron_nbr-1].weight[1]*(h-100)+(h-100)//10
            can.create_line(x_i4,y_i4,x_i14, y_i14,fill="blue")

def launch_SOM3():
    
    global nb_epoch, carte, epoch_time, nb_iter,can1,c, data
    
    if n.get() == "":
        n_data = 400
    else:
        n_data = int(n.get())
        
    if var.get() == "Square":
        data = square(n_data)
    elif var.get() == "Equireparted square":
        data = eq_square(n_data)
    elif var.get() == "Two Seperated circles":
        data = sep_circle(n_data)
    elif var.get() == "Circle":
        data = circle(n_data)
    elif var.get() == "Kind of Weights":
        data = weights(n_data)
        
    if neur.get() == "":
        neuron_nbr = 10
    else:
        neuron_nbr = int(neur.get())
        print(neuron_nbr)
    
    if nbiter.get() == "":
        nb_epoch = 300
        
    else:
        nb_epoch = int(nbiter.get())
        
    fen.destroy()
    fen1 = Tk()
    can1 = Canvas(fen1, width=w_l, height=h_l, bg='ivory')

    epoch_time = len(data)
    carte = SOM(data, star())
    
    nb_iter = epoch_time * nb_epoch
    #draw_SOM(neuron_nbr,carte,can,h_r,w_r)
    draw_map(neuron_nbr, data,carte, can1, h_l,w_l)
    def train10():
        for i in range(0,nb_iter+1):
            carte.train(i, epoch_time)
            if i%10 == 0 :
                can.delete("all")
                draw_map(neuron_nbr, data,carte, can1, h_l,w_l)
                can.update()
    def train100():
        for i in range(0,nb_iter+1):
            carte.train(i, epoch_time)
            if i%10 == 0 :
                can.delete("all")
                draw_map(neuron_nbr, data,carte, can1, h_l,w_l)
                can.update()
    can1.grid(row=1,rowspan=7,column=2)
    bou1 = Button(fen1, text='Quit', command=fen1.destroy)
    
    c=0
    def refresh():
        global c, carte, data, nb_iter
        if var1.get() == "end" :
            for i in range(0,nb_iter-c+1):
                carte.train(c+i, epoch_time)
            can1.delete("all")
            draw_map(neuron_nbr, data,carte, can1, h_l,w_l)
            can1.update()
            c = nb_iter
        else :
            N = int(var1.get())
            i = 0
            while i < N and c < nb_iter:
                if i % epoch_time == 0:
                    carte.generate_random_list()
                vect = carte.unique_random_vector()

                carte.train(c, epoch_time, vect)
                i += 1
                c = c+1
            can1.delete("all")
            draw_map(neuron_nbr, data,carte, can1, h_l,w_l)
            can1.update()

    
    def C_refresh():
        global c, carte, data, nb_iter
        if var1.get() == "end" :
            for i in range(0,nb_iter-c+1):
                if c+i % epoch_time == 0:
                    carte.generate_random_list()
                vect = carte.unique_random_vector()

                carte.train(c+i, epoch_time, vect)
            can1.delete("all")
            draw_map(neuron_nbr, data,carte, can1, h_l,w_l)
            can1.update()

        else:
            N = int(var1.get())
            for i in range (c, nb_iter-N,N):
                for k in range(1,N+1):
                    if i+k % epoch_time == 0:
                        carte.generate_random_list()
                    vect = carte.unique_random_vector()

                    carte.train(i+k,epoch_time, vect)
                can1.delete("all")
                draw_map(neuron_nbr, data,carte, can1, h_l,w_l)
                can1.update()

            c = nb_iter
                
    txt = Label(fen1, text = "Number of iteration?")

    var1 = StringVar()
    RB1 = Radiobutton(fen1,text="+1",value="1",variable=var1)
    RB2 = Radiobutton(fen1,text="+10",value="10",variable=var1)
    RB3 = Radiobutton(fen1,text="+100",value="100",variable=var1)
    RB4 = Radiobutton(fen1,text="+500",value="500",variable=var1)
    RB5 = Radiobutton(fen1,text="To the end",value="end",variable=var1)
    
    bou2 = Button(fen1, text='Step-by-step', command=refresh)
    bou3 = Button(fen1, text='Run', command=C_refresh)
    
    txt.grid(row=0)
    RB1.grid(row=1)
    RB2.grid(row=2)
    RB3.grid(row=3)
    RB4.grid(row=4)
    RB5.grid(row=5)
    bou2.grid(row=6)
    bou3.grid(row=7)
    bou1.grid(row=8)
    
    fen1.mainloop()

    
fen=Tk()
txt0 = Label(fen, text = "Name")
fic = Entry(fen)

txt1 = Label(fen, text = "Number of epoch")
nbiter = Entry(fen)

#On choisit quelle distribution de données on souhaite
txt2 = Label(fen, text = "Number of data :")
n = Entry(fen)
txt3 = Label(fen, text = "Number of neurons :")
neur = Entry(fen)
bou = Button(fen, text="Launch the SOM", command=launch_SOM3)

#txt0.grid(row=0)
txt1.grid(row=1)
txt2.grid(row=2)
txt3.grid(row=3)

#fic.grid(row=0,column=1)
nbiter.grid(row=1,column=1)
n.grid(row=2,column=1)
neur.grid(row=3,column=1)

txt4 = Label(fen, text = "What kind of data?")
txt4.grid(row =4)
var = StringVar()

rb1 = Radiobutton(fen,text="Square",value="Square",variable=var)
rb2 = Radiobutton(fen,text="Equireparted square",value="Equireparted square",variable=var)
rb3 = Radiobutton(fen,text="Two Seperated circles",value="Two Seperated circles",variable=var)
rb4 = Radiobutton(fen,text="Circle",value="Circle",variable=var)
rb5 = Radiobutton(fen,text="Kind of Weights",value="Kind of Weights",variable=var)


rb1.grid(row=5)
rb2.grid(row=6)
rb3.grid(row=7)
rb4.grid(row=8)
rb5.grid(row=9)

bou.grid(row=10)

fen.mainloop()

