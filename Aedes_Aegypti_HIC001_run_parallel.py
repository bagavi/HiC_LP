import imp, sys
import numpy as np
import utils, algorithms, run_parallel


chromosome = sys.argv[1]
no_process = int(sys.argv[2])
no_exps = int(sys.argv[3])

points = np.loadtxt('data/Aedes_aegypti/scaffold_positions/scaffold_positions_chr'+chromosome)
data_file = 'data/Aedes_aegypti/HIC001/chromosomes/chr'+chromosome+'.data'
folder = 'data/Aedes_aegypti/HIC001/error_rates/'

#Variable scaffolds
print ("Running experiments (SG, MG, BP, LP) for variable length contigs")
run_parallel.parallel_variable(chromosome, points, data_file, folder, no_process=32, no_exps=1)



print ("Running experiments (SG, MG, BP, LP) for 1000k length contigs")
run_parallel.parallel_100k(chromosome, points, data_file, folder, no_process=32, no_exps=1)