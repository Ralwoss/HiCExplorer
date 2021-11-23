import warnings
import argparse
import cooler as cool
import numpy as np
import tensorflow as tf
from math import sqrt

from hicexplorer.parserCommon import CustomFormatter
from hicexplorer._version import __version__

import logging

log = logging.getLogger(__name__)

warnings.simplefilter(action="ignore", category=RuntimeWarning)
warnings.simplefilter(action="ignore", category=PendingDeprecationWarning)


def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=CustomFormatter,
        add_help=False,
        conflict_handler='resolve',
        description="""
Uses a neural network trained by hicBuildNeuralNetwork to predict TAD boundaries

    $ hicNeuralNetworkPrediction --matrix hic_matrix.cool --binsize 10000 --model path/to/model
"""
    )

    parserRequired = parser.add_argument_group('Required arguments')

    parserRequired.add_argument('--matrix', '-m',
                                help='HiCExplorer matrix in cooler format.',
                                required=True)

    parserRequired.add_argument('--binsize',
                                help='bin size of the Hi-C matrix',
                                type=int,
                                required=True)

    parserRequired.add_argument('--model',
                                help='Path under which the trained model is found.',
                                required=True)

    parserRequired.add_argument('--indexShiftValue',
                                help='Number of bins that the window will be shifted to the right when constructing '
                                     'inputs for the neural network. Ideally this is the same value as the center size '
                                     'to ensure search coverage of the whole genome',
                                type=int,
                                required=True)

    parserOpt = parser.add_argument_group('Optional arguments')

    parserOpt.add_argument('--chromosomes', '-c',
                           help='The Chromosomes you want to predict the TAD boundaries for. If no Chromosomes '
                                'are given, TADs are predicted for the whole genome.',
                           nargs='+',
                           default=None,
                           required=False)

    parserOpt.add_argument('--out', '-o',
                           help='file where the TAD boundaries will be written to',
                           default="./boundaries.bed")

    return parser


def main(args=None):
    args = parse_arguments().parse_args(args)

    model = tf.keras.models.load_model(args.model)
    c = cool.Cooler(args.matrix)
    m = c.matrix(balance=False)
    shift = args.indexShiftValue

    window_size = int(sqrt(model.get_config()["layers"][0]["config"]["batch_input_shape"][1]))
    if args.chromosomes is None:
        chroms = c.chromnames
    else:
        chroms = args.chromosomes

    with open(args.out, "w") as o:
        o.write("chrom\tchromStart\tchromEnd\n")
        for ch in chroms:
            print(f"Working on Chromosome {ch}")
            b1 = None
            b2 = None
            ext = c.extent(ch)
            start = ext[0]
            end = ext[1]
            sm_start = start
            sm_end = sm_start + window_size
            while sm_end <= end:
                sm = np.array([m[sm_start:sm_end, sm_start:sm_end].flatten()])
                pred = model.predict(sm)
                if pred > 0.5:
                    if b1 is None:
                        b1 = sm_start + (window_size - 1) / 2 - start
                    elif b2 is None:
                        b2 = sm_start + (window_size - 1) / 2 - start
                        o.write(f"{ch}\t{int(b1*args.binsize)}\t{int(b2*args.binsize)}\n")
                    else:
                        b1, b2 = b2, sm_start + (window_size - 1) / 2 - start
                        o.write(f"{ch}\t{int(b1*args.binsize)}\t{int(b2*args.binsize)}\n")
                sm_start += shift
                sm_end += shift





    return
