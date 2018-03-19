import imp, sys
import numpy as np
import utils, algorithms, run_parallel


chromosome = sys.argv[1]
no_process = int(sys.argv[2])
no_exps = int(sys.argv[3])

points = np.loadtxt('data/HomoSapiens/scaffold_positions/scaffold_positions_chr'+chromosome)
data_file = 'data/HomoSapiens/chromosomes/chr'+chromosome+'.data'
folder = 'experiments/HomoSapiens/error_rates/'

#Variable scaffolds
print ("Running experiments (SG, MG, BP, LP) for variable length contigs")
run_parallel.parallel_variable(chromosome, points, data_file, folder, no_process=no_process, no_exps=no_exps)


print ("Running experiments (SG, MG, BP, LP) for 1000k length contigs")
run_parallel.parallel_100k(chromosome, points, data_file, folder, no_process=no_process, no_exps=no_exps)