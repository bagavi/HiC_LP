import imp, sys
import numpy as np
import utils, algorithms, run_parallel


chromosome = sys.argv[1]
no_process = int(sys.argv[2])
no_exps = int(sys.argv[3])

points = np.loadtxt('data/Gasterosteus/scaffold_positions/scaffold_positions_chr'+chromosome)
data_file = 'data/Gasterosteus/chromosomes/chr'+chromosome+'.data'
folder = 'real_dataset_experiments/Gasterosteus/error_rates/'

#Variable scaffolds
if chromosome!="1":
    print ("Running experiments (SG, MG, BP, LP) for variable length contigs")
    run_parallel.parallel_variable(chromosome, points, data_file, folder, no_process=no_process, no_exps=no_exps)


print ("Running experiments (SG, MG, BP, LP) for 1000k length contigs")
run_parallel.parallel_100k(chromosome, points, data_file, folder, no_process=no_process, no_exps=no_exps)