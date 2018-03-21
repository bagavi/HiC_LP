import math,csv, time
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import linalg as LA
from multiprocessing import Pool, Array, Process, Queue
import datetime, pickle

import imp, sys, utils
from gurobipy import *


plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def find_nearest_idx(array,value):
    tmp = array-value
    try:
        idx = np.where(tmp>=0)[0][0]
    except:
        idx = array.shape[0] - 1
    return idx

def Gaussian_matrix(mu,n=1000):
    M = np.random.normal(size=(n,n), scale=np.sqrt(0.5))
    M = M + M.T 
    M[np.diag_indices(n)] = 0
    M[utils.kth_diag_indices(M,1)] += mu
    M[utils.kth_diag_indices(M,n-1)] += mu
    M[utils.kth_diag_indices(M,-1)] += mu
    M[utils.kth_diag_indices(M,-n+1)] += mu
    return M

def Measurement_matrix_variable(data_file,points, sampling_frac=1, columns=['Pos1', 'Pos2', 'qua']):
    df = pd.DataFrame.from_csv(data_file, sep='\t', header=None, index_col=None)
    #print "Number of reads", df.shape
    df.columns =columns
    pos_array = df[['Pos1', 'Pos2']].as_matrix()
    n = points.shape[0]
    M_full = np.zeros((n,n))
    L = pos_array.shape[0]
    np.random.shuffle(pos_array)
    for i, row in enumerate(pos_array[:int(L*sampling_frac)]):
        #if i%500000 ==0:
        #    print i*100/L, 
        x = find_nearest_idx(points, row[0])
        y = find_nearest_idx(points, row[1])
        M_full[x][y] += 1
    M_full = M_full + M_full.T
    return M_full


def Measurement_matrix_equal(data_file,length, sampling_frac=1, columns=['Pos1', 'Pos2', 'qua']):
    df = pd.DataFrame.from_csv(data_file, sep='\t', header=None, index_col=None)
#    print "Number of reads", df.shape
    df.columns =columns
    pos_array = df[['Pos1', 'Pos2']].as_matrix()
    n = max(pos_array[:,0].max(), pos_array[:,1].max())/length+1
    M_full = np.zeros((n,n))
    L = pos_array.shape[0]
    np.random.shuffle(pos_array)
    for i, row in enumerate(pos_array[:int(L*sampling_frac)]):
#        if i%42000 ==0:
#            print i*100/L, 
        x = int(math.floor(row[0]/length))
        y = int(math.floor(row[1]/length))
        M_full[x][y] += 1
    M_full = M_full + M_full.T
    return M_full
   


"""
    Returns top 2k neighbhouring nodes based only on E_ij.
    Expected to perform decently
"""

def correct_data_answer(data,k=1):
    n,_ = data.shape
    correct_answer = np.zeros_like(data)
    answer = np.zeros_like(data)

    #Correct answer
    for i in range(1,k+1):
        correct_answer[utils.kth_diag_indices(correct_answer, k=n-i)] = 1
        correct_answer[utils.kth_diag_indices(correct_answer, k=i-n)] = 1
        correct_answer[utils.kth_diag_indices(correct_answer, k=i)]   = 1
        correct_answer[utils.kth_diag_indices(correct_answer, k=-i)]  = 1
    return correct_answer


def naive_greedy(data, file, sampling_frac=1):
    np.fill_diagonal(data,0)
    n,_ = data.shape
    answer = np.zeros_like(data)
    # Greedy
    nearest_neighbhour =  np.argsort(-1*data)[:,:2]
    for i in range(n):
        answer[i][nearest_neighbhour[i]] = 1
    #Correct answer
    correct_answer = correct_data_answer(data,k=1)
    
    error = (np.where(correct_answer-answer!=0)[0].shape[0]+0.0)/(2*n)
    fp = (np.where(correct_answer-answer==-1)[0].shape[0]+0.0)/(2*n)
    if file != None:
        fd = open(file,'a')
        fd.write(', '.join([str(error), str(fp) ,str(sampling_frac),"\n"]))
        fd.close()
    #print "SG", error, fp
    algo = "Naive"
    exptype = file.split('/')[-1].split('_')[0]
    now = datetime.datetime.now()
    folder = '/'.join(file.split('/')[:-1])+'/experiments/'
    filename = "__"+str(sampling_frac)+"_"+algo
    timestamp = now.strftime("%T_%M_%d_%Y")
    pos = file.find("greedy")
    chr = file[pos+6:-4]
    filename = folder+timestamp+filename+"_"+chr+"_"+exptype+".pkl"
    with open(filename,'wb') as f:
        pickle.dump([chr, sampling_frac, n, np.where(answer==1) ] ,f)
    
    return answer, error, fp

def greedy_but_cautious(data, file, sampling_frac=1):
    np.fill_diagonal(data,0)
    n,_ = data.shape
    answer = np.zeros_like(data)

    # Greedy but cautious
    count_arr = np.zeros(n)
    order = np.dstack(np.unravel_index(np.argsort(data.ravel()), data.shape))[0][::-1]
    assigned = 0
    for i, row in enumerate(order):
        if row[0] > row[1]: #each entry is present twice 
            continue
        if count_arr[row[0]] <2 and  count_arr[row[1]]<2:
            answer[row[0]][row[1]]=1
            answer[row[1]][row[0]]=1
            count_arr[row[0]]    +=1
            count_arr[row[1]]    +=1
            assigned +=1
        if assigned >n*2+1:
            break
    #Correct answer    
    correct_answer = correct_data_answer(data,k=1)
           
    error = (np.where(correct_answer-answer!=0)[0].shape[0]+0.0)/(2*n)
    fp = (np.where(correct_answer-answer==-1)[0].shape[0]+0.0)/(2*n)
    #print "MG", error, fp
    if file != None:
        fd = open(file,'a')
        fd.write(', '.join([str(error), str(fp) ,str(sampling_frac),"\n"]))
        fd.close()
    algo = "SmartGredy"
    now = datetime.datetime.now()
    exptype = file.split('/')[-1].split('_')[0]
    folder = '/'.join(file.split('/')[:-1])+'/experiments/'
    filename = "__"+str(sampling_frac)+"_"+algo
    timestamp = now.strftime("%T_%M_%d_%Y")
    pos = file.find("greedy")
    chr = file[pos+6:-4]
    filename = folder+timestamp+filename+"_"+chr+"_"+exptype+".pkl"
    with open(filename,'wb') as f:
        pickle.dump([chr, sampling_frac, n, np.where(answer==1) ] ,f)

def Solve_LP(data):
    time1 = time.time()
    model = Model("lp")
    model.params.LogToConsole = 0 #No output
    n = data.shape[0]
#     print  data.shape[0],greedy_but_cautious
    X = model.addVars(n, n, name='X', lb=0, ub=1, vtype=GRB.CONTINUOUS)

    model.addConstrs((X.sum(i,'*') == 2.0
                     for i in range(n)), name='R')
    model.addConstrs((X.sum('*',j) == 2.0
                     for j in range(n)), name='C')

    model.addConstrs((X.sum(i,i) <= 0
                     for i in range(n)), name='xxB')

    model.addConstrs((X[j,i] == X[i,j]
                     for i in range(n) for j in range(i)), name='symmetry')

    model.setObjective(sum(X[i,j]*data[i,j]
                      for i in range(n) for j in range(n)), GRB.MAXIMIZE)

    model.optimize()
    return model, time.time() - time1


def LP(data, file, sampling_frac=1):
    n = data.shape[0]
    model , t      = Solve_LP(data)
    answer         = np.array(model.X).reshape(n,n)
    correct_answer = correct_data_answer(data,k=1)
           
    error = (np.where(correct_answer-answer!=0)[0].shape[0]+0.0)/(2*n)
    fp = (np.where(correct_answer-answer==-1)[0].shape[0]+0.0)/(2*n)
    print "LP", sampling_frac, error, fp
    if file != None:
        fd = open(file,'a')
        fd.write(', '.join([str(error), str(fp) ,str(sampling_frac),"\n"]))
        fd.close()
        
    algo = "LP"
    now = datetime.datetime.now()
    exptype = file.split('/')[-1].split('_')[0]
    folder = '/'.join(file.split('/')[:-1])+'/experiments/'
    filename = "__"+str(sampling_frac)+"_"+algo
    timestamp = now.strftime("%T_%M_%d_%Y")
    pos = file.find("greedy")
    chr = file[pos+6:-4]
    filename = folder+timestamp+filename+"_"+chr+"_"+exptype+".pkl"
    with open(filename,'wb') as f:
        pickle.dump([chr, sampling_frac, n, np.where(answer==1), np.where(answer==0.5) ] ,f)
        
def Solve_BP(M, n_steps):
    n = M.shape[0]
    BP_matrix = np.zeros((2, n, n)) #BP_matrix[i] will store the state of BP in step i
    BP_matrix[0] = M.copy() #First state of BP_matrix is the data matrix itself
    accuracy = np.zeros(n_steps) #Stores accuracy after each step
    
    #True answer
    true_answer = np.zeros((n,2))
    true_answer[:,0] = np.roll(np.arange(n),1)
    true_answer[:,1] = np.roll(np.arange(n),-1)
    true_answer = true_answer.astype(int)

    for l in range(1, n_steps):
        for i in range(n):
            for j in range(n):
                row = BP_matrix[(l-1)% 2][:,i].copy()
                row[j] = np.inf
                BP_matrix[l%2][i][j] = M[i][j] - np.partition(row, 1)[1]
        BP_matrix[l%2][np.diag_indices_from(BP_matrix[l%2])] = np.inf
        

        if accuracy[l] == 1:
            accuracy[l:] = accuracy[l]
        if l > 50 and abs(np.mean(accuracy[l-25:l]) - np.mean(accuracy[l-50:l-25])) < 0.001: 
            accuracy[l:] = np.mean(accuracy[l-25:l])
            break
    M_final = BP_matrix[l%2]
    M_final[np.diag_indices_from(M_final)] = np.inf
    topTwo = M_final.argsort(axis=1)[:,:2]
    answer = np.zeros((n,n))
    for i in range(n):
        answer[i][topTwo[i]] =1
    return answer

def BP(data, file, sampling_frac=1):
    n = data.shape[0]
    answer     = Solve_BP(-data, n*10)
    correct_answer = correct_data_answer(data,k=1)
           
    error = (np.where(correct_answer-answer!=0)[0].shape[0]+0.0)/(2*n)
    fp = (np.where(correct_answer-answer==-1)[0].shape[0]+0.0)/(2*n)
    print "BP", sampling_frac, error, fp
    if file != None:
        fd = open(file,'a')
        fd.write(', '.join([str(error), str(fp) ,str(sampling_frac),"\n"]))
        fd.close()
        
    algo = "BP"
    now = datetime.datetime.now()
    exptype = file.split('/')[-1].split('_')[0]
    folder = '/'.join(file.split('/')[:-1])+'/experiments/'
    filename = "__"+str(sampling_frac)+"_"+algo
    timestamp = now.strftime("%T_%M_%d_%Y")
    pos = file.find("greedy")
    chr = file[pos+6:-4]
    filename = folder+timestamp+filename+"_"+chr+"_"+exptype+".pkl"
    with open(filename,'wb') as f:
        pickle.dump([chr, sampling_frac, n, np.where(answer==1), np.where(answer==0.5) ] ,f)