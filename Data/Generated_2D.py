import numpy as np


# Random data in [0,1]²
def square(n):
    data = np.array([np.random.random(2) for i in range(n)])
    return data


# Random distribution following a fractal shape.
# Has disjointed elements.
def sierpinski_carpet(n, level=2):
    data = np.array([np.random.random(2) for i in range(n)])
    for i in range(n):
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
