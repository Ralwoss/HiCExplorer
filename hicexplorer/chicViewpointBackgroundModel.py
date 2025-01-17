import argparse
import math
from multiprocessing import Process, Queue
import time
import logging
log = logging.getLogger(__name__)

import numpy as np
import fit_nbinom

from hicmatrix import HiCMatrix as hm
from hicexplorer._version import __version__
from .lib import Viewpoint


def parse_arguments(args=None):

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
        description="""
chicViewpointBackgroundModel computes a background model for all given samples with all reference points. For all relative distances to a reference point
a negative binomial distribution is fitted. In addition, for each relative distance to a reference point the average value for this location is computed. Both
background models are used, the first one for p-value and significance computation, the second one to filter out interactions with a smaller x-fold over the mean.

The background distributions are fixed at `--fixateRange`, i.e. all distances lower or higher than this value use the fixed background distribution.

An example usage is:

$ chicViewpointBackgroundModel --matrices matrix1.cool matrix2.cool matrix3.cool --referencePoints referencePointsFile.bed --range 20000 40000 --outFileName background_model.bed
"""
    )

    parserRequired = parser.add_argument_group('Required arguments')

    parserRequired.add_argument('--matrices', '-m',
                                help='The input matrices (samples) to build the background model on.',
                                nargs='+',
                                required=True)

    parserRequired.add_argument('--referencePoints', '-rp',
                                help='Bed file contains all reference points which should be used to build the background model.',
                                type=str,
                                required=True)

    parserOpt = parser.add_argument_group('Optional arguments')
    parserOpt.add_argument('--averageContactBin',
                           help='Average the contacts of n bins via a sliding window approach'
                           ' (Default: %(default)s).',
                           type=int,
                           default=5)
    parserOpt.add_argument('--truncateZeros', '-tz',
                           help='Truncates the zeros before the distributions are fitted. Use it in case you observe an over dispersion.',
                           required=False,
                           action='store_true')
    parserOpt.add_argument('--outFileName', '-o',
                           help='The name of the background model file'
                           ' (Default: %(default)s).',
                           default='background_model.txt')
    parserOpt.add_argument('--threads', '-t',
                           help='Number of threads (uses the python multiprocessing module)'
                           ' (Default: %(default)s).',
                           required=False,
                           default=4,
                           type=int
                           )
    parserOpt.add_argument('--fixateRange', '-fs',
                           help='Fixate score of backgroundmodel starting at distance x. E.g. all values greater 500kb are set to the value of the 500kb bin'
                           ' (Default: %(default)s).',
                           required=False,
                           default=500000,
                           type=int
                           )
    parserOpt.add_argument('--help', '-h', action='help',
                           help='show this help message and exit')

    parserOpt.add_argument('--version', action='version',
                           version='%(prog)s {}'.format(__version__))

    return parser


def compute_background(pReferencePoints, pViewpointObj, pArgs, pQueue):

    background_model_data = {}
    relative_positions = set()
    try:
        for i, referencePoint in enumerate(pReferencePoints):

            region_start, region_end, _ = pViewpointObj.calculateViewpointRange(
                referencePoint, (pArgs.fixateRange, pArgs.fixateRange))

            data_list, _ = pViewpointObj.computeViewpoint(
                referencePoint, referencePoint[0], region_start, region_end)

            # set data in relation to viewpoint, upstream are negative values, downstream positive, zero is viewpoint
            view_point_start, _ = pViewpointObj.getReferencePointAsMatrixIndices(
                referencePoint)
            view_point_range_start, view_point_range_end = \
                pViewpointObj.getViewpointRangeAsMatrixIndices(
                    referencePoint[0], region_start, region_end)

            for i, data in zip(range(view_point_range_start, view_point_range_end, 1), data_list):
                relative_position = i - view_point_start

            if pArgs.averageContactBin > 0:
                data_list = pViewpointObj.smoothInteractionValues(
                    data_list, pArgs.averageContactBin)

            for i, data in zip(range(view_point_range_start, view_point_range_end, 1), data_list):
                relative_position = i - view_point_start
                if relative_position in background_model_data:
                    background_model_data[relative_position].append(data)
                else:
                    background_model_data[relative_position] = [data]
                    relative_positions.add(relative_position)
    except Exception as exp:
        pQueue.put('Fail: ' + str(exp))
        return
    pQueue.put([background_model_data, relative_positions])
    return


def main(args=None):
    args = parse_arguments().parse_args(args)

    viewpointObj = Viewpoint()
    referencePoints, _ = viewpointObj.readReferencePointFile(
        args.referencePoints)

    relative_positions = set()
    bin_size = 0

    # - compute for each condition (matrix):
    # - all viewpoints and smooth them: sliding window approach
    # - after smoothing, sum all viewpoints up to one
    # - compute the percentage of each position with respect to the total interaction count
    # for models of all conditions:
    # - compute nbinom parameters

    referencePointsPerThread = len(referencePoints) // args.threads
    queue = [None] * args.threads
    process = [None] * args.threads
    background_model_data = None
    fail_flag = False
    fail_message = ''

    for matrix in args.matrices:
        hic_ma = hm.hiCMatrix(matrix)
        viewpointObj.hicMatrix = hic_ma

        bin_size = hic_ma.getBinSize()
        all_data_collected = False
        thread_done = [False] * args.threads
        for i in range(args.threads):

            if i < args.threads - 1:
                referencePointsThread = referencePoints[i * referencePointsPerThread:(i + 1) * referencePointsPerThread]
            else:
                referencePointsThread = referencePoints[i * referencePointsPerThread:]

            queue[i] = Queue()
            process[i] = Process(target=compute_background, kwargs=dict(
                pReferencePoints=referencePointsThread,
                pViewpointObj=viewpointObj,
                pArgs=args,
                pQueue=queue[i]
            )
            )

            process[i].start()

        while not all_data_collected:
            for i in range(args.threads):
                if queue[i] is not None and not queue[i].empty():
                    background_data_thread = queue[i].get()
                    if 'Fail:' in background_data_thread:
                        fail_flag = True
                        fail_message = background_data_thread[6:]
                        queue[i] = None
                        process[i].join()
                        process[i].terminate()
                        process[i] = None
                        thread_done[i] = True
                        continue
                    background_model_data_thread, relative_positions_thread = background_data_thread
                    if background_model_data is None:
                        background_model_data = background_model_data_thread
                    else:
                        for relativePosition in background_model_data_thread:
                            if relativePosition in background_model_data:
                                background_model_data[relativePosition].extend(
                                    background_model_data_thread[relativePosition])
                            else:
                                background_model_data[relativePosition] = background_model_data_thread[relativePosition]

                    relative_positions = relative_positions.union(
                        relative_positions_thread)
                    queue[i] = None
                    process[i].join()
                    process[i].terminate()
                    process[i] = None
                    thread_done[i] = True
            all_data_collected = True
            for thread in thread_done:
                if not thread:
                    all_data_collected = False
            time.sleep(1)

        del hic_ma
        del viewpointObj.hicMatrix

    if fail_flag:
        log.error('An error occurred caused by one or many faulty reference points.')
        log.error('Please run chicQualityControl to remove these from your reference point file: {}'.format(args.referencePoints))
        log.error(fail_message)
        exit(1)
    # for models of all conditions:
    # - fit negative binomial for each relative distance
    relative_positions = sorted(relative_positions)
    nbinom_parameters = {}
    max_value = {}
    mean_value = {}
    sum_all_values = 0
    data_of_distribution = None
    for relative_position in relative_positions:

        if args.truncateZeros:
            data_of_distribution = np.array(background_model_data[relative_position])
            mask = data_of_distribution > 0.0
            data_of_distribution = data_of_distribution[mask]
        else:
            data_of_distribution = np.array(background_model_data[relative_position])
        nbinom_parameters[relative_position] = fit_nbinom.fit(data_of_distribution)

        if len(data_of_distribution) > 0:
            max_value[relative_position] = np.max(data_of_distribution)
            average_value = np.average(data_of_distribution)
            mean_value[relative_position] = average_value
            sum_all_values += average_value
        else:
            max_value[relative_position] = 0.0
            average_value = 0.0
            mean_value[relative_position] = 0.0
            sum_all_values += 0.0

    for relative_position in relative_positions:
        mean_value[relative_position] /= sum_all_values
    # write result to file
    with open(args.outFileName, 'w') as file:
        file.write(
            'Relative position\tsize nbinom\tprob nbinom\tmax value\tmean value\n')

        for relative_position in relative_positions:
            relative_position_in_genomic_scale = relative_position * bin_size
            file.write("{}\t{:.12f}\t{:.12f}\t{:.12f}\t{:.12f}\n".format(relative_position_in_genomic_scale, nbinom_parameters[relative_position]['size'],
                                                                         nbinom_parameters[relative_position]['prob'], max_value[relative_position], mean_value[relative_position]))
