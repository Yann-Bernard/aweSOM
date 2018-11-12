from Models.SOM import *
np.set_printoptions(threshold=np.inf)

def kohonen_model(n):
    
    M=[[0 for i in range (n)] for j in range (n)]
    
    V = [0,0,0,0,1]
    nulle = [0,0,0,0,0]
    
    mat_haut = [nulle]
    for i in range(1,4):
        mat_haut.append(V)
    mat_haut.append([0,1,1,1,0])
    
    mat_centre=[]    
    for i in range(4):
        mat_centre.append(V)
    mat_centre.append([1,1,1,1,0])
    
    mat_bas=[]
    for i in range(4):
        mat_bas.append(V)
    mat_bas.append([1,1,1,0,0])
    
    mat_gauche = [V,nulle,V,V,[1,0,1,1,0]]
    
    mat_droite = [V,V,nulle,V,[1,1,0,1,0]]
    
    
    for i in range(n):
        for j in range(n):
            if i == 0:
                if j == 0:
                    M[i][j] = [nulle,V,nulle,V,[0,1,0,1,0]]
                elif 0 < j < n-1:
                    M[i][j] = mat_haut 
                elif j == n-1:
                    M[i][j] = [nulle,nulle,V,V,[0,0,1,1,0]]

            elif 0<i<n-1:
                if j == 0:
                    M[i][j] = mat_gauche
                elif 0 < j < n-1:
                    M[i][j] = mat_centre
                elif j == n-1:
                    M[i][j] = mat_droite

            elif i == n-1:
                if j == 0:
                    M[i][j] = [V,V,nulle,nulle,[1,1,0,0,0]]
                elif 0 < j < n-1:
                    M[i][j] = mat_bas
                elif j == n-1:
                    M[i][j] = [V,nulle,V,nulle,[1,0,1,0,0]]
    return np.array(M)

def snake(n):
    M=[[0 for i in range (n)] for j in range (n)]
    nulle=[0,0,0,0,0]
    A=[1,0,0,0,1]
    B=[0,0,0,1,1]
    
    for i in range(n):
        for j in range(n):
            if i == 0:
                if j%2 == 0:
                    M[i][j] = [nulle,nulle,B,[0,0,1,0,1],[0,0,1,1,0]]
                else:
                    M[i][j] = [nulle,B,nulle,[0,1,0,0,1],[0,1,0,1,0]]
            elif i == n-1:
                if j%2 == 0:
                    M[i][j] = [[0,0,1,0,1],nulle,nulle,A,[1,0,0,1,0]]
                else:
                    M[i][j] = [[0,1,0,0,1],A,nulle,nulle,[1,1,0,0,0]]
            else:
                M[i][j] = [B,nulle,nulle,A, [1,0,0,1,0]]
    
    return np.array(M)
            
                
n = 2

test_k = SOM(n, kohonen_model(n))

test_s = SOM(n, snake(n))

test_k.compute_neurons_distance()
test_s.compute_neurons_distance()

print("Distance de test avec une matrice type kohonen")
print(test_k.neural_dist)
#print(np.array(test_k.adj))
#print(np.array(test_k.MDist))
print('\n')
print("Distance de test avec une matrice type ligne")
#print(np.array(test_s.MDist))
print(test_s.neural_dist)
    
    
    

    
    
        