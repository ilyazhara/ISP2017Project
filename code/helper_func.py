import pandas as pd
import numpy as np
from itertools import product

def savearray(q, n, arr, filename):
    f = open(filename, 'w')
    for idx in product(range(n), repeat=q):
            f.write(str(np.array(idx)) + ' ')
            for elem in arr[idx]:
                f.write(str(elem) + ' ')
            f.write('\n')
    f.close() 

    

def saveall(JP_pred, JE_pred, UP_pred, UE_pred):
    JP_pred = JP_pred.full()
    JE_pred = JE_pred.full()
    UP_pred = UP_pred.full().astype(int)
    UE_pred = UE_pred.full().astype(int)
    savearray(3, 11, JP_pred, 'with_tt_jp.txt')
    savearray(3, 11, JE_pred, 'with_tt_je.txt')
    savearray(4, 11, UP_pred, 'with_tt_up.txt')
    savearray(4, 11, UE_pred, 'with_tt_ue.txt')


def get_neighbors(idx, sizes, up, ue, mode):
    up = up.astype(int)
    ue = ue.astype(int)
    neighbors = list()
    pos_list = list()
    if mode == 'p':
        z = list(idx)
        z[2] += ue[0]
        z[3] += ue[1]
        for positions in product(up_x, up_y):
            if ((idx[0] + positions[0] >= 0) and (idx[0] + positions[0] < sizes[0]) and 
                (idx[1] + positions[1] >= 0) and (idx[1] + positions[1] < sizes[1])):
                z[0] = idx[0] + positions[0]
                z[1] = idx[1] + positions[1]
                neighbors.append(tuple(z))
                pos_list.append(np.array(positions))
    if mode == 'e':
        z = list(idx)
        z[0] += up[0]
        z[1] += up[1]
        for positions in product(ue_x, ue_y):
            if ((idx[2] + positions[0] >= 0) and (idx[2] + positions[0] < sizes[2]) and 
                (idx[3] + positions[1] >= 0) and (idx[3] + positions[1] < sizes[3])):
                z[2] = idx[2] + positions[0]
                z[3] = idx[3] + positions[1]
                neighbors.append(tuple(z))
                pos_list.append(np.array(positions))
                
    return neighbors, pos_list


def get_data(from_, to_, n):
    x_grid = np.linspace(from_, to_, n)
    y_grid = np.linspace(from_, to_, n)
    
    return x_grid, y_grid


def solve(neighbors, pos_list, G, J_pred, mode, gamma=0.7):
    function_vals = np.zeros(len(neighbors))
    for i, neighbor in enumerate(neighbors):
        function_vals[i] = G[neighbor] + gamma * J_pred[neighbor]
    if mode == 'p':
        idx = np.argmin(function_vals)
        val = function_vals[idx]
        pos = pos_list[idx]
    elif mode == 'e':
        idx = np.argmax(function_vals)
        val = function_vals[idx]
        pos = pos_list[idx]
    else:
        print 'Mode is not correct!'
    
    return val, pos


def tt_cross_P(V):
    result = np.zeros((V.shape[0], 3))
    for i in xrange(V.shape[0]):
        idx = (int(V[i, 0]), int(V[i, 1]), int(V[i, 2]), int(V[i, 3]))
        idx1 = (int(V[i, 0]), int(V[i, 1]), int(V[i, 2]), int(V[i, 3]), 0)
        idx2 = (int(V[i, 0]), int(V[i, 1]), int(V[i, 2]), int(V[i, 3]), 1)
        neighbors, pos_list = get_neighbors(idx, sizes, up=np.zeros(2), 
                                            ue=np.array([UE_pred[idx1], UE_pred[idx2]]), mode='p')
        value, positions = solve(neighbors, pos_list, G, JP_pred, mode='p')
        result[i, 0] = value
        result[i, 1] = positions[0]
        result[i, 2] = positions[1]
        
    return result


def tt_cross_E(V):
    result = np.zeros((V.shape[0], 3))
    for i in xrange(V.shape[0]):
        idx = (int(V[i, 0]), int(V[i, 1]), int(V[i, 2]), int(V[i, 3]))
        idx1 = (int(V[i, 0]), int(V[i, 1]), int(V[i, 2]), int(V[i, 3]), 0)
        idx2 = (int(V[i, 0]), int(V[i, 1]), int(V[i, 2]), int(V[i, 3]), 1)
        neighbors, pos_list = get_neighbors(idx, sizes, 
                                            up=np.array([UP_pred[idx1], UP_pred[idx2]]), ue=np.zeros(2), mode='e')
        value, positions = solve(neighbors, pos_list, G, JE_pred, mode='e')
        result[i, 0] = value
        result[i, 1] = positions[0]
        result[i, 2] = positions[1]
        
    return result