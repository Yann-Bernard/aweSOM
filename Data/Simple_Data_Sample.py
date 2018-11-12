import numpy as np
#------ Données random dans [0,1]
def square(n):
    data = np.array([np.random.random(2) for i in range(n)])
    return data
def eq_square(n):
        n=int(np.sqrt(n))
        data=np.array([np.array([i,j]) for i in range (n) for j in range(n)])/n #test avec une equirépartition des stimuli
        return data

#------- On prend des données en paquets distincts
def sep_circle(n):
    n = n//4
    o1=np.array([3,4])
    r1=2
    o2=np.array([15,10])
    r2=5
    
    data1=[]
    data2=[]
    
    for i in range(500):
        p=np.random.random()
        if p>0.5:
            a1=o1[0]+r1*np.random.random()
            a2=o2[0]+r2*np.random.random()
    
        else:
            a1=o1[0]-r1*np.random.random()
            a2=o2[0]-r2*np.random.random()
            
        pprime=np.random.random()
        if pprime>0.5:
            b1=o1[1]+np.sqrt(r1**2-(a1-o1[0])**2)*np.random.random()
            b2=o2[1]+np.sqrt(r2**2-(a2-o2[0])**2)*np.random.random()
            
        else:
            b1=o1[1]-np.sqrt(r1**2-(a1-o1[0])**2)*np.random.random()
            b2=o2[1]-np.sqrt(r2**2-(a2-o2[0])**2)*np.random.random()
        data1.append([a1,b1])
        data2.append([a2,b2])
        
        
    data=np.array(data1+data2)
    
    return data

#----- On prend 2 paquets de données reliées par une barres
def weights(n):
    n = n//5
    o1=np.array([3,4])
    o2=np.array([21,4])
    r=4
    
    
    data1=[]
    data2=[]
    data3=[]
    
    for i in range(n):
        p=np.random.random()
        if p>0.5:
            a1=o1[0]+r*np.random.random()
            a2=o2[0]+r*np.random.random()
    
        else:
            a1=o1[0]-r*np.random.random()
            a2=o2[0]-r*np.random.random()
            
        pprime=np.random.random()
        if pprime>0.5:
            b1=o1[1]+np.sqrt(r**2-(a1-o1[0])**2)*np.random.random()
            b2=o2[1]+np.sqrt(r**2-(a2-o2[0])**2)*np.random.random()
            
        else:
            b1=o1[1]-np.sqrt(r**2-(a1-o1[0])**2)*np.random.random()
            b2=o2[1]-np.sqrt(r**2-(a2-o2[0])**2)*np.random.random()
        psec=np.random.random()
        
        a3=(o1[0]+(r/2)*np.sqrt(3))+(o2[0]-o1[0]-r*np.sqrt(3))*np.random.random()
        
        if psec>0.5:
            b3=o1[1]+(r/2)*np.random.random()
        else:
            b3=o1[1]-(r/2)*np.random.random()
        data1.append([a1,b1])
        data2.append([a2,b2])
        data3.append([a3,b3])
    data=np.array(data1+data2+data3)
    return data

#--- Données en cercle
def circle(n):
    n = n//2
    data=[]
    for i in range(n):
        p=np.random.random()
        if p>0.5:
            a1=o1[0]+r*np.random.random()
        elif p<0.5:
            a1=o1[0]-r*np.random.random()
        pprime=np.random.random()
        if pprime>0.5:
            b1=o1[1]+np.sqrt(r**2-(a1-o1[0])**2)*np.random.random()
        elif pprime<0.5:
            b1=o1[1]-np.sqrt(r**2-(a1-o1[0])**2)*np.random.random()
    
        data.append([a1,b1])
    
    data=np.array(data)
    return data


def sierpinski_carpet(number, level=4):
    data = np.array([np.random.random(2) for i in range(number)])
    for i in range(number):
        correct = False
        while not correct:
            j = 0
            data[i] = np.random.random(2)
            x = data[i][0]
            y = data[i][1]
            while j < level and not correct:
                j += 1
                if 1/3 < x < 2/3 and 1/3 < y < 2/3:
                    correct = True
                else:
                    x = (x % (1/3)) * 3
                    y = (y % (1/3)) * 3
    return data


# def complicated_sierpinski_carpet(number, level=3):
#     data = np.array([np.random.random(2) for i in range(number)])
#     area = np.zeros(level)
#     for i in range(level):
#         area[i] = 8**i/9**(i+1)
#     area = np.divide(area,sum(area))
#     for i in range(number):
#         j = 0
#         position = np.random.random()
#         data[i] = np.divide(data[i], 3)+1/3
#         while position > area[j]:
#             position -= area[j]
#             j += 1
#     return data
