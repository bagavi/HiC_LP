import math,csv, random, collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import scipy.sparse as sparse
from scipy.optimize import curve_fit
from scipy.misc import factorial
import pickle
from iced.normalization import ICE_normalization
from multiprocessing import Process, Queue
from multiprocessing import Pool
from collections import Counter
import imp, sys
from numpy import linalg as LA
import utils, algorithms


def parallel_variable(chromosome, points, data_file, folder, no_process=32, no_exps=5):
    sampling_frac_array   = np.array([ 0.05,  0.1,  0.15,  0.2,  0.25, 0.35 , 0.45 , 0.6, 0.85,   1.])
    simple_greedy_file = folder+"variable_simple_greedy"+chromosome+".csv"
    merge_greedy_file  = folder+"variable_merge_greedy"+chromosome+".csv"
    LP_greedy_file     = folder+"variable_LP_stricter_greedy"+chromosome+".csv"
    BP_greedy_file     = folder+"variable_BP_stricter_greedy"+chromosome+".csv"
    print "Launching", no_process, "jobs"
    pool = Pool(processes=no_process)  
    processes = []
    for i, sampling_frac in enumerate(sampling_frac_array):
        print "\n", sampling_frac, "\t",
        for j in range(no_exps):
            print j, "\t",
            Corr_matrix = algorithms.Measurement_matrix_variable(data_file, points, sampling_frac, 
                                                                 columns=['chr', 'Pos1', 'Pos2'])

            Corr_matrix[np.diag_indices_from(Corr_matrix)] = 0
            Corr_matrix, bias = ICE_normalization(Corr_matrix, eps=1e-4, max_iter=1000, output_bias=True)

            #Simple Greedy
            args = (Corr_matrix, simple_greedy_file, sampling_frac)
            p = Process(target = algorithms.naive_greedy, args = args)
            p.start()
            processes.append(p)


            #Merge Greedy
            args = (Corr_matrix, merge_greedy_file, sampling_frac)
            p = Process(target = algorithms.greedy_but_cautious, args = args)
            p.start()
            processes.append(p)

            #LP
            args = (Corr_matrix, LP_greedy_file, sampling_frac)
            p = Process(target = algorithms.LP, args = args)
            p.start()
            processes.append(p)

            #BP
            args = (Corr_matrix, BP_greedy_file, sampling_frac)
            p = Process(target = algorithms.BP, args = args)
            p.start()
            processes.append(p)

    print  "\nJoining Process"
    for i, p in enumerate(processes):
        print i, "\t",
        p.join()
        
        
def parallel_100k(chromosome, points, data_file, folder, no_process=32):
    length = 100000
    sampling_frac_array1   = np.array([ 0.05,  0.1,  0.15,  0.2,  0.25, 0.35 , 0.45 , 0.6, 0.85,   1.])/3.0
    sampling_frac_array2   = np.array([ 0.05,  0.1,  0.15,  0.2,  0.25, 0.35 , 0.45 , 0.6, 0.85,   1.])
    sampling_frac_array = np.concatenate([sampling_frac_array1, sampling_frac_array2])
    simple_greedy_file = folder+"100000_simple_greedy_chr"+chromosome+".csv"
    merge_greedy_file  = folder+"100000_merge_greedy_chr"+chromosome+".csv"
    LP_greedy_file     = folder+"100000_LP_greedy_chr"+chromosome+".csv"
    BP_greedy_file     = folder+"100000_BP_greedy_chr"+chromosome+".csv"

    no_exps = 10
    no_process = no_exps*3*sampling_frac_array.shape[0]
    print "Launching", no_process, "jobs"
    pool = Pool(processes=no_process)  
    processes = []
    for i, sampling_frac in enumerate(sampling_frac_array):
        print "\n", sampling_frac, "\t",
    #     algorithms.naive_greedy(Corr_matrix)
        for j in range(no_exps):
            print j, "\t",
            Corr_matrix               = algorithms.Measurement_matrix_equal(data_file, length, sampling_frac, 
                                                                 columns=['chr', 'Pos1', 'Pos2'])
            Corr_matrix = Corr_matrix[:1000, :1000]
            Corr_matrix[np.diag_indices_from(Corr_matrix)] = 0
            Corr_matrix, bias = ICE_normalization(Corr_matrix, eps=1e-4, max_iter=1000, output_bias=True)

            #Simple Greedy
            args = (Corr_matrix, simple_greedy_file, sampling_frac)
            p = Process(target = algorithms.naive_greedy, args = args)
            p.start()
            processes.append(p)


            #Merge Greedy
            args = (Corr_matrix, merge_greedy_file, sampling_frac)
            p = Process(target = algorithms.greedy_but_cautious, args = args)
            p.start()
            processes.append(p)

            #LP
            args = (Corr_matrix, LP_greedy_file, sampling_frac)
            p = Process(target = algorithms.LP, args = args)
            p.start()
            processes.append(p)    

            #BP
            args = (Corr_matrix, BP_greedy_file, sampling_frac)
            p = Process(target = algorithms.BP, args = args)
            p.start()
            processes.append(p)    

    print  "\nJoining Process"
    for i, p in enumerate(processes):
        print i, "\t",
        p.join()