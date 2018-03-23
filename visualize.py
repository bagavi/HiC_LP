import math,csv, random, collections, pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('dark')
current_palette = sns.color_palette()



def return_results(filepath):
    f = open(filepath, 'rb')
    reader = csv.reader(f)
    results = []
    for row in reader:
        results += [row[:-1]]
    results = np.array(results)
    f.close()
    sampling_frac_array = np.unique(results[:,-1])
    greedy_error_array    = np.zeros_like(sampling_frac_array, dtype='float')
    greedy_error_count_array    = np.zeros_like(sampling_frac_array, dtype='float')
    greedy_fp_array = np.zeros_like(sampling_frac_array, dtype='float')
    greedy_fp_count_array = np.zeros_like(sampling_frac_array, dtype='float')
    for row in results:
        error, fp, frac = row
        pos = np.where(sampling_frac_array==frac)[0][0]
        greedy_error_array[pos] += float(error)
        greedy_error_count_array[pos] += 1
        greedy_fp_array[pos] += float(fp)
        greedy_fp_count_array[pos] += 1
    error_array = greedy_error_array/greedy_error_count_array
    fp_array    = greedy_fp_array/greedy_fp_count_array
    return sampling_frac_array, error_array, fp_array

def plot(simple_greedy_file, merge_greedy_file, lp_greedy_file, bp_greedy_file):
    sg_sf_array, sg_error_array, sg_fp_array = return_results(simple_greedy_file)
    mg_sf_array, mg_error_array, mg_fp_array = return_results(merge_greedy_file)
    lp_sf_array, lp_error_array, lp_fp_array = return_results(lp_greedy_file)
    bp_sf_array, bp_error_array, bp_fp_array = return_results(bp_greedy_file)
    print lp_fp_array, lp_error_array/2.
    plt.figure(figsize=(20,6))
    plt.plot(sg_sf_array, sg_error_array/6+sg_fp_array*(2.0/3.0), '--', lw = 2, label="Greedy", 
             color = current_palette[0], alpha = 0.7)
    plt.plot(mg_sf_array, mg_error_array/6+mg_fp_array*(2.0/3.0), '--', lw = 2, label="Merge", 
             color = current_palette[1], alpha = 0.7)
    plt.plot(lp_sf_array, lp_error_array/6+lp_fp_array*(2.0/3.0), lw = 2, label="f-LP", 
             color = 'black', alpha = 0.8)
    plt.plot(bp_sf_array, bp_error_array/6+bp_fp_array*(2.0/3.0), lw = 2, label="BP", 
             color = 'red', alpha = 0.8)

    try:
        plt.axvline(np.log(stats['n'])/stats['signal_mean'], label= 'IT limit')
    except:
        pass
    plt.xlabel("Sampling fraction", fontsize=20)
    plt.ylabel("Error", fontsize=20)
    plt.ylim((0,.8))
    plt.legend(fontsize=16)
    
def plotsemilogy(simple_greedy_file, merge_greedy_file, lp_greedy_file):
    sg_sf_array, sg_error_array, sg_fp_array = return_results(simple_greedy_file)
    mg_sf_array, mg_error_array, mg_fp_array = return_results(merge_greedy_file)
    lp_sf_array, lp_error_array, lp_fp_array = return_results(lp_greedy_file)
    plt.figure(figsize=(20,6))
#     plt.semilogy(sg_sf_array, sg_error_array, label="SG: total error")
    plt.semilogy(sg_sf_array, sg_fp_array, label="SG")
#     plt.semilogy(mg_sf_array, mg_error_array, label="MG: total error")
    plt.semilogy(mg_sf_array, mg_fp_array, label="MG")
#     plt.semilogy(lp_sf_array, lp_error_array, lw=3, label="LP: total error")
    plt.semilogy(lp_sf_array, lp_fp_array, lw=3, label="LP: False")
    plt.xlabel("sampling frac")
    plt.ylabel("error")
    plt.legend()

def plotvariable(folder, chromosome):
    simple_greedy_file = folder+"variable_simple_greedy"+chromosome+".csv"
    merge_greedy_file  = folder+"variable_merge_greedy"+chromosome+".csv"
    LP_greedy_file     = folder+"variable_LP_stricter_greedy"+chromosome+".csv"
    BP_greedy_file     = folder+"variable_BP_stricter_greedy"+chromosome+".csv"
    plot(simple_greedy_file, merge_greedy_file, LP_greedy_file, BP_greedy_file)
    plt.title("Variable contigs of chromosome "+chromosome, fontsize=25)
    plt.show()
    
def plot_100k(folder, chromosome, xlim=1):
    simple_greedy_file = folder+"100000_simple_greedy_chr"+chromosome+".csv"
    merge_greedy_file  = folder+"100000_merge_greedy_chr"+chromosome+".csv"
    LP_greedy_file     = folder+"100000_LP_greedy_chr"+chromosome+".csv"
    BP_greedy_file     = folder+"100000_BP_greedy_chr"+chromosome+".csv"
    plot(simple_greedy_file, merge_greedy_file, LP_greedy_file, BP_greedy_file)
    plt.xlim(0,xlim)
    plt.title("Variable contigs of chromosome "+chromosome)
    plt.show()