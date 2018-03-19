#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import erf
import matplotlib.mlab as mlab
import scipy.integrate as integrate
import matplotlib.patches as mpatches
import pylab as P
from scipy.optimize import minimize
from statsmodels.sandbox.stats.multicomp import multipletests
from collections import Counter
from scipy import signal
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import pandas as pd
from scipy.optimize import curve_fit
from scipy.misc import factorial
import itertools

##Global variable

global map_scaffold
map_scaffold = {}

global map_index
map_index = 0

global scaf_size_array
scaf_size_array = []

global _bad_edge_weights
_bad_edge_weights = []

global RNM_matrix
RNM_matrix = []

global threshold_length
threshold_length = 100000


def data_stat(M):
    n = M.shape[0]
    signal = M[kth_diag_indices(M, 1)]
    noise = np.array([0])
    for i in range(2,n):
        diag_i = M[kth_diag_indices(M, i)]
        noise = np.concatenate((noise, diag_i))
    dict1 = {   "signal_mean": signal.mean(),
                "noise_mean": noise.mean(),
                "signal_std": signal.std(),
                "noise_std": noise.std(),
                 "n": n
            }
    return dict1


# Converts an 2d array to a dictionary

def timing(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print '%s function took %0.3f s' % (f.func_name, (time2-time1))
        return ret
    return wrap


def twodlist_dict(array):
    array_dict = dict()
    for i in range(len(array)):
        array_dict[array[i][1]] = array[i][0]
    return array_dict


# Plots the positions of array 1 present in array 2 too.

def array1_in_array2(array1, array2):
    for i in range(len(array2)):
        if i % 6000 == 0:
            plt.figure(figsize=(600, 5))
            plt.show()
        try:
            index = array2.index(array1[i])
            plt.axvspan(i, i + 1, color='yellow', alpha=0.5, lw=0)
        except:
            plt.axvspan(i, i + 1, color='white', alpha=0.5, lw=0)


#########################################################################################################
## Takes care of the indexing the scaffold names to number
#########################################################################################################

def scaffold_name_mapper(name):
    global map_scaffold
    global map_index
    if map_scaffold.has_key(name):
        return map_scaffold[name]
    else:
        map_scaffold[name] = map_index
        map_index += 1
        return map_index - 1


def scaffold_name_parser(name):
    global map_scaffold
    global map_index
    if name[2] == '0':
        name = name[:2] + name[3:]
    if name[2] == '0':
        name = name[:2] + name[3:]

    return map_scaffold[name]


##########################################################################################################

#########################################################################################################
# File Readers
#########################################################################################################

# Returns array with column 1 containing scaffold name and column two containing its length
@timing
def open_scaffold_report_human_ch22(
    filepath='../scaffold_report.csv',
    namepos=4,
    lenpos=1,
    truncate=0,
    ):
    print ('Reading file..', filepath)
    ifile = open(filepath, 'r')
    reader = csv.reader(ifile, delimiter=' ')
    global scaf_size_array
    global map_scaffold
    flag = False
    for row in reader:
        name = row[0].split('\t')[namepos]
        scaf_size_array += [[ int(row[1]),int(row[2]) ]]
        scaffold_name_mapper(name)
    scaf_size_array = np.array(scaf_size_array)
    return (scaf_size_array, map_scaffold)

@timing
def open_scaffold_report_scaf1(
    filepath='../scaffold_report.csv',
    namepos=0,
    lenpos=1,
    truncate=0,
    ):
    print ('Reading file..', filepath)
    ifile = open(filepath, 'r')
    reader = csv.reader(ifile, delimiter=' ')
    global scaf_size_array
    global map_scaffold
    flag = False
    for row in reader:
        row = row[0].split('\t')
        scaf_size_array += [[ int(row[0][2:-19]),int(row[1]) ]]
    scaf_size_array = np.array(scaf_size_array)
    return (scaf_size_array, map_scaffold)

@timing
def open_scaffold_report(
    filepath='../scaffold_report.csv',
    namepos=4,
    lenpos=1,
    truncate=0,
    ):
    print ('Reading file..', filepath)
    ifile = open(filepath, 'r')
    reader = csv.reader(ifile, delimiter=' ')
    global scaf_size_array
    global map_scaffold
    flag = False
    for row in reader:
        name = row[0].split('\t')[namepos]
        try:
            length = int(row[1].split('\t')[lenpos])
        except:
            length = int(row[0].split('\t')[lenpos])

        # Bad code

        if truncate != 0:
            name = name[:truncate]

        scaf_size_array += [[scaffold_name_mapper(name), length]]
    scaf_size_array = np.array(scaf_size_array)
    print ('Finished reading file..', filepath)
    return (scaf_size_array, map_scaffold)


def cross_read_score(row, lower_limit):
    global scaf_size_array
    global threshold_length

    scaf_1_pos = int(row[3])
    scaf_2_pos = int(row[4])

    scaf_1_length = scaf_size_array[map_scaffold[row[0]]][1]
    scaf_2_length = scaf_size_array[map_scaffold[row[1]]][1]

    read_quality = float(row[2])
    orientation  = 0

    if read_quality < lower_limit:
        return False, orientation

    # The read is originating from middle of the scaffold -
    # It is covering a distance greater than the threshold_length and hence its
    # very likely to be noisy read

    # Distance from the endpoints of scaf1
    if scaf_1_pos < scaf_1_length - scaf_1_pos:
        gap1 = scaf_1_pos
        orientation += 1
    else:
        gap1 = scaf_1_length - scaf_1_pos
    
    # Distance from the endpoints of scaf1
    if scaf_2_pos < scaf_2_length - scaf_2_pos:
        gap2 = scaf_2_pos
        orientation += 10
    else:
        gap2 = scaf_2_length - scaf_2_pos

    # Gap of the cross reads is greater than gap1 + gap 2

    if gap1 + gap2 > threshold_length:
        return 0, orientation
    return threshold_length - (gap1 + gap2), orientation


def cross_read_score_new(row, lower_limit):
    global scaf_size_array
    global threshold_length

    scaf_1_pos = int(row[3])
    scaf_2_pos = int(row[4])

    scaf_1_length = scaf_size_array[int(row[0])][1]
    scaf_2_length = scaf_size_array[int(row[1])][1]

    read_quality = float(row[2])
    orientation  = 0

    if read_quality < lower_limit:
        return False, orientation

    # The read is originating from middle of the scaffold -
    # It is covering a distance greater than the threshold_length and hence its
    # very likely to be noisy read

    # Distance from the endpoints of scaf1
    if scaf_1_pos < scaf_1_length - scaf_1_pos:
        gap1 = scaf_1_pos
        orientation += 1
    else:
        gap1 = scaf_1_length - scaf_1_pos
    
    # Distance from the endpoints of scaf1
    if scaf_2_pos < scaf_2_length - scaf_2_pos:
        gap2 = scaf_2_pos
        orientation += 10
    else:
        gap2 = scaf_2_length - scaf_2_pos

    # Gap of the cross reads is greater than gap1 + gap 2

    if gap1 + gap2 > threshold_length:
        return 0, orientation
    return threshold_length - (gap1 + gap2), orientation

@timing
def cross_reads_filereader_new(filepath, lower_limit=10):
    count = 0
    print ('Reading file..', filepath)
    print ('Discarding reads below the quality score of ', lower_limit)
    ifile = open(filepath, 'r')
    reader = csv.reader(ifile, delimiter=' ')
    crossread_locations = []
    pos = 0
    for row in reader:
        pos += 1
        if pos % 2000000 == 0:
            print pos, count
        score, orientation = cross_read_score_new(row, lower_limit)
        if score!=0 :
            count += 1
            code = orientation%10 + 2*(orientation/10)
            crossread_locations.append([ int(row[0]), \
                                         int(row[1]), \
                                         int(row[2]), \
                                         int(row[3]), \
                                         int(row[4]), \
                                         score, \
                                         code \
                                       ])
    print ('Finshed reading file..', filepath)
    print ('Total number of cross_reads', count)
    return crossread_locations


# Reads cross-reads from the filepath and returns 'crossread_locations
@timing
def cross_reads_filereader(filepath, lower_limit=10):
    count = 0
    print ('Reading file..', filepath)
    print ('Discarding reads below the quality score of ', lower_limit)
    ifile = open(filepath, 'r')
    reader = csv.reader(ifile, delimiter=' ')
    crossread_locations = []
    pos = 0
    for row in reader:
        pos += 1
        if pos % 2000000 == 0:
            print pos, count
        score, orientation = cross_read_score(row, lower_limit)
        if score!=0 :
            count += 1
            code = orientation%10 + 2*(orientation/10)
            crossread_locations.append([ scaffold_name_parser(row[0]), \
                                         scaffold_name_parser(row[1]), \
                                         score/100, \
                                         code \
                                       ])
    print ('Finshed reading file..', filepath)
    print ('Total number of cross_reads', count)
    return crossread_locations


# Reads same-reads from the filepath and returns sameread_locations
@timing
def cross_reads_filereader_new_human_ch22(filepath, lower_limit=10):
    count = 0
    print ('Reading file..', filepath)
    print ('Discarding reads below the quality score of ', lower_limit)
    ifile = open(filepath, 'r')
    reader = csv.reader(ifile, delimiter=' ')
    crossread_locations = []
    pos = 0
    for row in reader:
        pos += 1
        if pos % 2000000 == 0:
            print pos, count
        score, orientation = cross_read_score_new(row, lower_limit)
        if score!=0 :
            count += 1
            crossread_locations.append([ int(row[0]), \
                                         int(row[1]), \
                                         int(row[2]), \
                                         int(row[3]), \
                                         int(row[4]), \
                                       ])
    print ('Finshed reading file..', filepath)
    print ('Total number of cross_reads', count)
    return crossread_locations


# Reads cross-reads from the filepath and returns 'crossread_locations

def same_reads_filereader(filepath, lower_limit=10):
    print ('Reading file..', filepath)
    print ('Discarding reads below the quality scaore of ', lower_limit)
    ifile = open(filepath, 'r')
    reader = csv.reader(ifile, delimiter=' ')
    sameread_locations = []
    for row in reader:
        if float(row[1]) >= lower_limit:
            sameread_locations += [scaffold_name_parser(row[0])]
    sameread_locations = np.array(sameread_locations)
    print ('Finished reading file..', filepath)
    print ('Toal number of same_reads', sameread_locations.shape[0])

    return sameread_locations


#########################################################################################################
# Returns
# 1) Array with crossread counting
# 2) Measurement Matrix
# 3) Identity Measurement Matrix (M'_ij = 1, if M_ij>0)

@timing
def crossread_computation(crossread_locations):
    global scaf_size_array
    print 'Preparing the measurement matrix'
                                        
    measurement_matrix = np.zeros([scaf_size_array.shape[0],
                                  scaf_size_array.shape[0], 4])
    count = 0
    for row in crossread_locations:
        count += 1
        first_scaf = int(row[0])
        second_scaf = int(row[1])
        orientation = int(row[3])
        measurement_matrix[first_scaf, second_scaf, orientation] +=  float(row[2])
        if orientation == 2:
            orientation = 1
        elif orientation == 1:
            orientation = 2
        measurement_matrix[second_scaf, first_scaf, orientation] += float(row[2])


    # Converting crosscount array into suitable form
    return (measurement_matrix)

@timing
def crossread_computation_new(crossread_locations):
    global scaf_size_array
    print 'Preparing the measurement matrix'
                                        
    measurement_matrix = np.zeros([scaf_size_array.shape[0],
                                  scaf_size_array.shape[0], 4])
    count = 0
    for row in crossread_locations:
        count += 1
        first_scaf = int(row[0])
        second_scaf = int(row[1])
        orientation = int(row[-1])
        measurement_matrix[first_scaf, second_scaf, orientation] += float(row[-2])
        if orientation == 2:
            orientation = 1
        elif orientation == 1:
            orientation = 2
        measurement_matrix[second_scaf, first_scaf, orientation] += float(row[-2])


    # Converting crosscount array into suitable form
    return (measurement_matrix)

@timing
def crossread_computation_human_ch22(n,crossread_locations):
    global scaf_size_array
    print 'Preparing the measurement matrix'
                                        
    measurement_matrix = np.zeros([n,n])
    count = 0
    for row in crossread_locations:
        count += 1
        first_scaf = int(row[0])
        second_scaf = int(row[1])
        measurement_matrix[first_scaf, second_scaf] += 1


    # Converting crosscount array into suitable form
    return (measurement_matrix+measurement_matrix.T)


# Returns counting of read frequencies - USed for plotting

@timing
def counting_read_frequencies(crossread_locations, sameread_locations):
    print 'Cross read frequency counting'

    # Cross-reads

    crossread_locations = np.array(crossread_locations)  # Converting array to np.array
    crossread_location_pos1_counter = Counter(crossread_locations[:, 1])  # Counting the number of times the first position

                                                                        # appears in the cross-reads
    # Coverting the count values to np array and sorting them thereafter.

    crossread_countarray = \
        np.array(crossread_location_pos1_counter.most_common(),
                 dtype='int64')
    crossread_countarray = \
        crossread_countarray[np.argsort(crossread_countarray[:, 0])]

    print 'Same read frequency counting'

    # Same-reads

    sameread_locations_counter = Counter(sameread_locations[:, 0])
    sameread_countarray = \
        np.array(sameread_locations_counter.most_common(), dtype='int64')
    sameread_countarray = \
        sameread_countarray[np.argsort(sameread_countarray[:, 0])]

    print 'All read frequency counting'

    # All reads

    allreads_counter = sameread_locations_counter \
        + crossread_location_pos1_counter
    allreads_counter_array = np.array(allreads_counter.most_common())
    allreads_counter_array = \
        allreads_counter_array[np.argsort(allreads_counter_array[:, 0])]

    return (crossread_countarray, sameread_countarray,
            allreads_counter_array)


#########################################################################################################

# Plots xxxx-read of over lengts

def read_over_length(readarrays=[], ylabels=[]):
    global scaf_size_array
    reads_over_lengths = []
    scaf_size_array_dict = twodlist_dict(scaf_size_array)
    plt.figure(figsize=(12, 18))
    index = 0
    for (readarray, ylabel) in zip(readarrays, ylabels):
        reads_over_length = np.zeros_like(readarray, dtype='float64')
        reads_over_length[np.argsort(readarray[:, 0])]
        for i in range(reads_over_length.shape[0]):
            try:
                reads_over_length[i][0] = readarray[i][0] + 0.0
                reads_over_length[i][1] = float((readarray[i][1] + 0.0)
                        / (scaf_size_array_dict[readarray[i][0]] + 0.0))
            except:
                pass
        reads_over_lengths += [reads_over_length]
        in2 = np.ones(50) / 50
        x = reads_over_length[:, 0]
        y = signal.convolve(reads_over_length[:, 1], in2, mode='same')

        index += 1
        plt.subplot(str(len(readarrays)) + '1' + str(index))
        plt.semilogy(x, reads_over_length[:, 1], color='green')
        plt.semilogy(x, y, color='black')
        plt.xlabel('Scaffold index')
        plt.ylabel(ylabel)
    return reads_over_lengths


@timing
def plot_various_graphs(crossread_countarray, sameread_countarray,
                        allread_countarray):
    global scaf_size_array
    scaf_size_array = np.array(scaf_size_array, dtype='int32')
    print 'Number of scaffolds', scaf_size_array.shape[0]
    plt.figure(figsize=(20, 20))
    plt.subplots_adjust(hspace=0.4)
    plt.subplot(321)
    plt.hist(scaf_size_array[:,0], log=True,bins=50)
    plt.xlabel('Scaffold length')
    plt.ylabel('Number of scaffolds')
    plt.title('Bining Scaffold lengths')

    plt.subplot(322)
    plt.semilogy(scaf_size_array[:, 1], scaf_size_array[:, 0],
                 color='green')
    plt.xlabel('Scaffold index')
    plt.ylabel('Scaffold length')
    plt.title('Scaffold_length vs Index')

    plt.subplot(323)
    plt.semilogy(crossread_countarray[:, 0], crossread_countarray[:, 1])
    plt.semilogy(crossread_countarray[:, 0],
                 signal.convolve(crossread_countarray[:, 1],
                 np.ones(20) / 20, mode='same'))
    plt.xlabel('Scaffold index (decreasing order of size)')
    plt.ylabel('Number of cross-reads')
    plt.title('Number of cross-reads vs Index')

    plt.subplot(324)
    plt.semilogy(sameread_countarray[:, 0], sameread_countarray[:, 1])
    plt.semilogy(sameread_countarray[:, 0],
                 signal.convolve(sameread_countarray[:, 1], np.ones(20)
                 / 20, mode='same'))
    plt.xlabel('Scaffold index (decreasing order of size)')
    plt.ylabel('Number of same-reads')
    plt.title('Number of same-reads vs Index')

    plt.subplot(325)
    plt.semilogy(allread_countarray[:, 0], allread_countarray[:, 1])
    plt.semilogy(allread_countarray[:, 0],
                 signal.convolve(allread_countarray[:, 1], np.ones(20)
                 / 20, mode='same'))
    plt.ylabel('Number of cross-reads')
    plt.title('Number of all-reads vs Index')

    plt.show()


### Plots scaffolds with missing cross,same,all reads

@timing
def scaffolds_with_missing_reads(read_arrays, ylabels):
    global scaf_size_array
    scaf_size_set = set(np.arange(len(scaf_size_array)))
    index = 0
    for (read_array, ylabel) in zip(read_arrays, ylabels):
        read_set = set(read_array[:, 0])
        diff_list = np.array(list(scaf_size_set - read_set))
        if len(diff_list) == 0:
            print "No missing in", ylabel
            continue
        index += 1
        plt.subplot(str(len(read_arrays)) + '1' + str(index))

        plt.hist(diff_list)
        plt.xlabel('scaffold position (position = 1/ length)')
        plt.ylabel(ylabel)
        plt.title(str(diff_list.shape[0]))


def random_shuffle_matrix(matrix):
    perm = np.random.permutation(np.arange(matrix.shape[0]))
    return matrix.T[perm].T[perm]


def random_shuffle_matrix_2(matrix):
    perm = np.random.permutation(np.arange(matrix.shape[0]))
    return matrix.T[perm].T[perm], perm


# Plots the Matrix

def plot_matrix(
    matrix,
    cmap='flag',
    figsize=(12, 8),
    savepath='orig.png',
    clip_per=95,
    clip_to=0,
    title='colorMap',
    ):
    matrix = clip_array(matrix,clip_per)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.set_title(title)
    plt.imshow(matrix, cmap=cmap)
    plt.show()
    fig.savefig(savepath, dpi=fig.dpi)


# Saving the measurement matrix for further computation

@timing
def save_corr_matrix(filepath, measurement_matrix):
    filepath = 'Correlation_matrix/' + filepath
    print 'Saving the correlation matrix'
    measurement_matrix = np.array(measurement_matrix, dtype='int32')
    A = np.where(measurement_matrix > 0)
    Corr_matrix = np.zeros([A[0].shape[0], 3], dtype='int32')
    for i in range(A[0].shape[0]):
        Corr_matrix[i][0] = A[0][i]
        Corr_matrix[i][1] = A[1][i]
        Corr_matrix[i][2] = measurement_matrix[A[0][i], A[1][i]]
    np.savetxt(filepath, Corr_matrix, delimiter=',')
    print 'Saved the correlation matrix'


#########################################################################################################
#########################################################################################################

def reads_from_diff_scaffolds(
    filename,
    diff_savepath,
    same_savepath,
    new=False,
    ):
    print ('Processing', filename)
    ifile = open(filename, 'r')
    reader = csv.reader(ifile)
    diff = []
    same = []
    temp = 0
    for line in reader:
        if line[0][0] != 'H':
            continue
        temp += 1
        row = ''
        for i in line:
            row += i
        data = row.split('\t')

#        if data[0] == data[1]:
#            same += [['SS'+data[0]+'_Simulated_Scaffold', data[4]]]
#        else:
#            diff += [['SS'+data[0]+'_Simulated_Scaffold', 'SS'+data[1]+'_Simulated_Scaffold', data[4]]]
#        print data
 #       break

        if new:
            if data[6] == '=':
                same += [[data[2], data[4]]]
            else:
                diff += [[data[2], data[6], data[4]]]
        else:
            if data[0] == data[4]:
                same += [[data[0] + '_Simulated_Scaffold', data[2]]]
            else:
                diff += [[data[0] + '_Simulated_Scaffold', data[4]
                         + '_Simulated_Scaffold', data[2]]]

    same = np.array(same)
    diff = np.array(diff)

#    print same
#    print diff

    print ('diffscafs saving in file', diff_savepath)
    np.savetxt(diff_savepath, diff, delimiter=' ', newline='\n',
               fmt='%s')

    print ('same scafs saving in file', same_savepath)
    np.savetxt(same_savepath, same, delimiter=' ', newline='\n',
               fmt='%s')


#########################################################################################################
#########################################################################################################

#########################################################################################################
#########################################################################################################
# Saving the sacffold_info matrix for further computation

@timing
def save_scaf_info_matrix(filepath, lines=-1):
    global scaf_size_array
    if lines != -1:
        scaf_size_array = scaf_size_array[:lines]
    filepath = 'scaf_info/' + filepath
    np.savetxt(filepath, scaf_size_array, delimiter=',')


# Splitting scaffold into 750-1000 parts

@timing
def make_write_sub_scaffolds(
    scaffold,
    sub_scaffold_filepath,
    sub_scaffold_sam_filepath,
    variable_length=True,
    number_scaffolds=1000,
    plot=True,
    ):

    # Split "GL582980.1" scaffold into sub-scaffolds

    sub_scaffolds_records = []
    sub_scaffolds_length = []
    if variable_length:
        print 'The scaffold lengths are variable'

        # The scaffold length pdf"

        sub_scaffolds_length = []
        index = 0
        while len(scaffold) > 0:
            sampled_length = np.random.randint(2000, 10000 * 5)
            sampled_length = np.random.randint(40, 250) ** 2
            index += 1
            sub_scaffolds_records += \
                [SeqRecord(Seq(scaffold[:sampled_length]), id='SS'
                 + str(index), description='Simulated_Scaffold',
                 name='GL582980.1')]
            sub_scaffolds_length += [sampled_length]
            scaffold = scaffold[sampled_length:]

        print 'Generated ', index, ' scaffolds.'
        if plot:
            plt.figure(figsize=(10, 5))
            plt.subplot(121)
            plt.semilogy(sorted(sub_scaffolds_length)[::-1])
            plt.xlabel('Sub-scaffold_index')
            plt.ylabel('Length')

            plt.subplot(122)
            plt.hist(sorted(sub_scaffolds_length)[::-1])
            plt.xlabel('Sub-scaffold_index')
            plt.ylabel('Length')
    else:

        print 'The scaffold lengths are same'
        scaf_len = int(len(scaffold) / number_scaffolds)
        print 'Length of scaffolds', scaf_len
        for i in range(number_scaffolds):
            sub_scaffold = scaffold[i * scaf_len:i * scaf_len
                + scaf_len]
            sub_scaffolds_records += [SeqRecord(Seq(sub_scaffold),
                    id='SS' + str(i + 1),
                    description='Simulated_Scaffold', name='GL582980.1'
                    )]

    print ('Sub scaffolds written in ', sub_scaffold_filepath)
    SeqIO.write(sub_scaffolds_records, sub_scaffold_filepath, 'fasta')

    print 'Writing the report in ', sub_scaffold_sam_filepath \
        + '/sub_scaffolds_report.csv'
    with open(sub_scaffold_sam_filepath + '/sub_scaffolds_report.csv',
              'w') as fnafile:
        for row in sub_scaffolds_records:
            fnafile.write(row.id + '_' + str(row.description) + '\t'
                          + str(len(row.seq)) + '\n')


#########################################################################################################
#########################################################################################################
@timing
def Measurement_matrix_from_file(
    last_index,
    filepath,
    scaf_size_array,
    normalization_power_1=0.5,
    normalization_power_2=0.5,
    ):
    global RNM_Matrix
    ifile = open(filepath, 'r')
    reader = csv.reader(ifile, delimiter=' ')
    Correlation_list = []
    for row in reader:
        row_val = row[0].split(',')
        if float(row_val[0]) >= last_index:
            continue
        elif float(row_val[1]) >= last_index:
            continue
        Correlation_list += [ [float(row_val[0]), float(row_val[1]), float(row_val[2])] ]
                             
    Correlation_list = np.array(Correlation_list, dtype='int32')

    Measurement_matrix = np.zeros( [last_index, last_index], dtype='float32')
                                  
    for i in range(Correlation_list.shape[0]):
        row = Correlation_list[i]
        Measurement_matrix[row[0]][row[1]] = row[2]

    # Keeping only first (0,last_index)

    Measurement_matrix = Measurement_matrix + Measurement_matrix.T
    row_sum = np.sum(Measurement_matrix, axis=1)
    print len(np.where(row_sum == 0)[0]), ' have zero cross_reads'
    one_row_sum = 1 / row_sum
    one_row_sum[np.where(one_row_sum == np.inf)] = 0
    one_row_sum[np.where(one_row_sum == np.nan)] = 0
    rownormalization_term = np.outer(np.power(one_row_sum,
            normalization_power_1), np.power(one_row_sum,
            normalization_power_2))
    Row_Normalized_measurement_matrix = Measurement_matrix \
        * rownormalization_term
    RNM_Matrix = Row_Normalized_measurement_matrix
    return (Measurement_matrix, [],
            Row_Normalized_measurement_matrix)


def diag_zero(Arr):
    for i in range(Arr.shape[0]):
        Arr[i][i] = 0
    return Arr


### Ordering the scaffolds

def SNR(G):
    signal_edges = 0
    noise_edges = 0
    for i in G.edges():
        if abs(i[0] - i[1]) == 1:
            signal_edges += 1
        else:
            noise_edges += 1
    print 'Total Edges', signal_edges + noise_edges
    print 'Signal points', signal_edges
    print 'Noisy points', noise_edges
    print 'SNR', signal_edges / (noise_edges + 0.0)


def is_path_ordered(path):
    global RNM_Matrix
    global _bad_edge_weights
    bad_orders = 0
    for i in range(len(path) - 1):
        v1 = path[i]
        v2 = path[i + 1]
        if abs(v1 - v2) != 1:
            _bad_edge_weights += [RNM_Matrix[v1][v2]]

#             print v1,v2

            bad_orders += 1

#     if bad_orders > 0:
#         print bad_orders

    return bad_orders


def missing_edges_graph(nodes, G):
    edges = G.edges()
    misses = 0
    for i in range(len(nodes) - 2):
        v0 = nodes[i]
        v1 = nodes[i + 1]
        edge_1 = (v0, v1)
        edge_2 = (v1, v0)
        try:
            _ = edges.index(edge_1)
        except:
            misses += 1
    return misses


def find_nearest(array, value):
    idx = np.abs(array - value).argmin()
    return array[idx]


def print_args(function):

    def wrapper(*args, **kwargs):
        print 'Arguments:', args, kwargs
        return function(*args, **kwargs)

    return wrapper


def npread_csv(filename):
    df = pd.read_csv(filename, sep=' ', header=None)
    array = np.array(df)
    return array

def kth_diag_indices(a, k):
    rowidx, colidx = np.diag_indices_from(a)
    colidx = colidx.copy()  # rowidx and colidx share the same buffer

    if k > 0:
        colidx += k
    else:
        rowidx -= k
    k = np.abs(k)

    return rowidx[:-k], colidx[:-k]

# poisson function, parameter lamb is the fit parameter
def poisson(k, lamb):
    return (lamb**k/factorial(k)) * np.exp(-lamb)
def gaussian(x,mu,sigma):
    return  1/(math.sqrt(2*math.pi)*sigma)*np.exp(-((x-mu)**2/(2*sigma*sigma)))
def exponential(x,llambda):
    return llambda*np.exp(-llambda*x)

def clip_array(array, clip_per):
    clip_at = np.percentile(array[np.where(array>0)], clip_per)
    array[np.where(array>clip_at)] = clip_at
    return array

def kendallTau(A, B):
    pairs = itertools.combinations(range(0, len(A)), 2)

    distance = 0

    for x, y in pairs:
        a = A[x] - A[y]
        b = B[x] - B[y]

        # if discordant (different signs)
        if (a * b < 0):
            distance += 1

    return min(distance,len(A)*(len(A)-1)/2 - distance)
			