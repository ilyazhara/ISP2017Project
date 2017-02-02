import pandas as pd
import numpy as np
from itertools import product

def VI(n=11, 
		   Delta_max=1e-2, 
		   delta_max=1e-2,
		   gamma=0.7,
		   K_max=50,
		   k_max=100,
		   from_=0,
		   to_=11)
	
	JP_pred = np.zeros((n, n, n, n))
	JE_pred = np.zeros((n, n, n, n))
	JP_next = np.zeros((n, n, n, n))
	JE_next = np.zeros((n, n, n, n))
	JP = np.zeros((n, n, n, n))
	JE = np.zeros((n, n, n, n))
	G = np.zeros((n, n, n, n))
	sizes = np.array([n, n, n, n]).astype(int)
	UP_pred = np.zeros((n, n, n, n, 2)).astype(int)
	UE_pred = np.zeros((n, n, n, n, 2)).astype(int)
	UP_next = np.zeros((n, n, n, n, 2)).astype(int)
	UE_next = np.zeros((n, n, n, n, 2)).astype(int)
	x_grid, y_grid = get_data(from_, to_, n)
	up_x = np.arange(-2, 3).astype(int)
	up_y = np.arange(-2, 3).astype(int)
	ue_x = np.arange(-1, 2).astype(int)
	ue_y = np.arange(-1, 2).astype(int)

	for idx, val in zip(product(range(n), repeat=4), 
						product(x_grid, y_grid, x_grid, y_grid)):
	    JP_pred[idx] = (val[0] - val[2]) ** 2 + (val[1] - val[3]) ** 2 
	    JE_pred[idx] = (val[0] - val[2]) ** 2 + (val[1] - val[3]) ** 2
	    G[idx] = (val[0] - val[2]) ** 2 + (val[1] - val[3]) ** 2 + 10

	f_r = open('without_tt_report.txt', 'w')
	JP = JP_pred.copy()
	JE = JE_pred.copy()
	Delta_p = Delta_max + 1
	Delta_e = Delta_max + 1
	K = 0
	while ((Delta_p > Delta_max) and (Delta_e > Delta_max) and (K < K_max)):
	    delta = delta_max + 1
	    k = 0
	    while ((delta > delta_max) and (k < k_max)):
	        for idx in product(range(n), repeat=4):
	            neighbors, pos_list = get_neighbors(idx, sizes, up=np.zeros(2), ue=UE_pred[idx], mode='p')
	            JP_next[idx], UP_next[idx] = solve(neighbors, pos_list, G, JP_pred, mode='p')
	        k += 1
	        delta = np.linalg.norm(JP_next - JP_pred) ** 2
	        JP_pred = JP_next.copy()
	        UP_pred = UP_next.copy()
	        f_r.write('P ' + str(k) + ' ' + str(delta) + '\n')
	    delta = delta_max + 1
	    k = 0
	    while ((delta > delta_max) and (k < k_max)):
	        for idx in product(range(n), repeat=4):
	            neighbors, pos_list = get_neighbors(idx, sizes, up=UP_pred[idx], ue=np.zeros(2), mode='e')
	            JE_next[idx], UE_next[idx] = solve(neighbors, pos_list, G, JE_pred, mode='e')
	        k += 1
	        delta = np.linalg.norm(JE_next - JE_pred) ** 2
	        JE_pred = JE_next.copy()
	        UE_pred = UE_next.copy()
	        f_r.write('E ' + str(k) + ' ' + str(delta) + '\n')
	    Delta_p = np.linalg.norm(JP - JP_pred) ** 2
	    Delta_e = np.linalg.norm(JE - JE_pred) ** 2
	    JE = JE_pred.copy()
	    JP = JP_pred.copy()
	    UE = UE_pred.copy()
	    UP = UP_pred.copy()
	    K += 1
	    f_r.write('K ' + str(K) + ' ' + str(Delta_p) + ' ' + str(Delta_e) + '\n')  
	f_r.close()


import pandas as pd
import numpy as np
from itertools import product

def TT_VI(n=11, 
		   Delta_max=1e-2, 
		   delta_max=1e-2,
		   gamma=0.7,
		   K_max=50,
		   k_max=100,
		   from_=0,
		   to_=11,
		   eps=1e-3)
	
	JP_pred = np.zeros((n, n, n, n))
	JE_pred = np.zeros((n, n, n, n))
	JP_next = np.zeros((n, n, n, n))
	JE_next = np.zeros((n, n, n, n))
	JP = np.zeros((n, n, n, n))
	JE = np.zeros((n, n, n, n))
	G = np.zeros((n, n, n, n))
	sizes = np.array([n, n, n, n]).astype(int)
	UP_pred = np.zeros((n, n, n, n, 2)).astype(int)
	UE_pred = np.zeros((n, n, n, n, 2)).astype(int)
	UP_next = np.zeros((n, n, n, n, 2)).astype(int)
	UE_next = np.zeros((n, n, n, n, 2)).astype(int)
	x_grid, y_grid = get_data(from_, to_, n)
	up_x = np.arange(-2, 3).astype(int)
	up_y = np.arange(-2, 3).astype(int)
	ue_x = np.arange(-1, 2).astype(int)
	ue_y = np.arange(-1, 2).astype(int)

	arr1 = np.arange(n)
	arr2 = np.arange(n)
	arr3 = np.arange(n)
	arr4 = np.arange(n)
	arr1_tt = tt.tensor(arr1)
	arr2_tt = tt.tensor(arr2)
	arr3_tt = tt.tensor(arr3)
	arr4_tt = tt.tensor(arr4)
	e1 = tt.tensor(np.ones(n))
	e2 = tt.tensor(np.ones(n))
	e3 = tt.tensor(np.ones(n))
	e4 = tt.tensor(np.ones(n))
	I1 = tt.kron(tt.kron(tt.kron(arr1_tt, e2), e3), e4)
	I2 = tt.kron(tt.kron(tt.kron(e1, arr2_tt), e3), e4)
	I3 = tt.kron(tt.kron(tt.kron(e1, e2), arr3_tt), e4)
	I4 = tt.kron(tt.kron(tt.kron(e1, e2), e3), arr4_tt)

	for idx, val in zip(product(range(n), repeat=4), 
						product(x_grid, y_grid, x_grid, y_grid)):
	    JP_pred[idx] = (val[0] - val[2]) ** 2 + (val[1] - val[3]) ** 2 
	    JE_pred[idx] = (val[0] - val[2]) ** 2 + (val[1] - val[3]) ** 2
	    G[idx] = (val[0] - val[2]) ** 2 + (val[1] - val[3]) ** 2 + 10

	f2_r = open('with_tt_report.txt', 'w')
	JP = JP_pred.copy()
	JE = JE_pred.copy()
	JP = tt.tensor(JP)
	JE = tt.tensor(JE)
	JP_pred = tt.tensor(JP_pred)
	UP_pred = tt.tensor(UP_pred)
	JE_pred = tt.tensor(JE_pred)
	UE_pred = tt.tensor(UE_pred)
	Delta_p = Delta_max + 1
	Delta_e = Delta_max + 1
	K = 0
	while ((Delta_p > Delta_max) and (Delta_e > Delta_max) and (K < K_max)):
	    delta = delta_max + 1
	    k = 0
	    while ((delta > delta_max) and (k < k_max)):
	        new_tensor1 = tt.multifuncrs2([I1, I2, I3, I4], tt_cross_P, eps=eps, d2=3, verb=0)
	        new_array1 = new_tensor1.full().reshape((-1), order='f').reshape((
	                n, n, n, n, 3), order='f').reshape((n, n, n, n, 3), order='c')
	        JP_next = tt.tensor(new_array1[:, :, :, :, 0])
	        UP_next = tt.tensor(new_array1[:, :, :, :, 1:])
	        k += 1
	        delta = (JP_next - JP_pred).norm() ** 2
	        JP_pred = JP_next.copy()
	        UP_pred = UP_next.copy()
	        f2_r.write('P ' + str(k) + ' ' + str(delta) + '\n')
	    delta = delta_max + 1
	    k = 0
	    while ((delta > delta_max) and (k < k_max)):
	        new_tensor2 = tt.multifuncrs2([I1, I2, I3, I4], tt_cross_E, 
	        	eps=1e-3, d2=3, verb=0)
	        new_array2 = new_tensor2.full().reshape((-1), order='f').reshape((
	                n, n, n, n, 3), order='f').reshape((n, n, n, n, 3), order='c')
	        JE_next = tt.tensor(new_array2[:, :, :, :, 0])
	        UE_next = tt.tensor(new_array2[:, :, :, :, 1:])
	        k += 1
	        delta = (JE_next - JE_pred).norm() ** 2
	        JE_pred = JE_next.copy()
	        UE_pred = UE_next.copy()
	        f2_r.write('E ' + str(k) + ' ' + str(delta) + '\n')
	    Delta_p = (JP - JP_pred).norm() ** 2
	    Delta_e = (JE - JE_pred).norm() ** 2
	    JE = JE_pred.copy()
	    JP = JP_pred.copy()
	    UE = UE_pred.copy()
	    UP = UP_pred.copy()
	    K += 1
	    f2_r.write('K ' + str(K) + ' ' + str(Delta_p) + ' ' + str(Delta_e) + '\n')  
	f2_r.close()