import warnings
warnings.simplefilter(action="ignore", category=RuntimeWarning)
warnings.simplefilter(action="ignore", category=PendingDeprecationWarning)
import argparse
from hicmatrix import HiCMatrix as hm
from hicexplorer._version import __version__
from hicexplorer.utilities import toString
from hicmatrix.HiCMatrix import check_cooler
import logging
log = logging.getLogger(__name__)

import pandas as pd
from pybedtools import BedTool
import numpy as np
import cooler


def parse_arguments(args=None):

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
        description="""
This script overlaps the loop locations with protein locations to determine the accuracy of the loop detection.
Loops need to have format as follows:

`chr start end chr start end`

The protein peaks need to be in narrowPeaks or broadPeak format.

A protein match is successfull if at the bin of the x and y location a protein peak is overlapped.
A bin is assumed to have a protein if one or more protein peaks falling within the bin region.
The value of the protein is not considered, only match or non-match.
""")

    parserRequired = parser.add_argument_group('Required arguments')

    parserRequired.add_argument('--data', '-d',
                                help='The loop file from hicDetectLoops. To use files from other sources, '
                                'please follow \'chr start end chr start end\' format.',
                                required=True)
    parserRequired.add_argument('--validationData', '-vd',
                                help='The data file to validate the given locations. Can be narrowPeak, broadPeak, cool, or 2D text.',
                                required=True)
    parserRequired.add_argument('--validationType', '-vt',
                                help='The type of the validation data. Can be bed, or cool format',
                                choices=['bed', 'cool'],
                                default='bed')
    parserRequired.add_argument('--method', '-m',
                                help='The method used (for the moment only loop is possible)'
                                ' (Default: %(default)s).',
                                choices=['loops'],
                                default='loops')
    parserRequired.add_argument('--resolution', '-r',
                                help='The used resolution of the Hi-C interaction matrix.',
                                required=True,
                                type=int)
    parserOpt = parser.add_argument_group('Optional arguments')
    parserOpt.add_argument('--outFileName', '-o',
                           help='The prefix name of the output files. Two file are written: output_matched_locations and output_statistics.'
                           'First file contains all loop locations with protein location matches, second file contains statistics about this matching.'
                           )
    parserOpt.add_argument('--chrPrefixLoops', '-cl',
                           help='Adding / removing / do nothing a \'chr\'-prefix to chromosome name of the loops.',
                           choices=[None, 'add', 'remove'],
                           default=None
                           )
    parserOpt.add_argument('--chrPrefixProtein', '-cp',
                           help='Adding / removing / do nothing a \'chr\'-prefix to chromosome name of the protein.',
                           choices=[None, 'add', 'remove'],
                           default=None
                           )

    parserOpt.add_argument('--help', '-h', action='help',
                           help='show this help message and exit')

    parserOpt.add_argument('--version', action='version',
                           version='%(prog)s {}'.format(__version__))

    return parser


def readProtein(pFile, pChrPrefixProtein):
    protein_df = pd.read_csv(pFile, sep='\t', header=None)[[0, 1, 2]]
    if pChrPrefixProtein == 'add':
        protein_df[0] = 'chr' + protein_df[0].astype(str)
    elif pChrPrefixProtein == 'remove':
        protein_df[0] = protein_df[0].str.lstrip('chr')
    return protein_df


def readLoopFile(pInputFile, pChrPrefixLoops):
    full_loop = pd.read_csv(pInputFile, sep='\t', header=None)
    if pChrPrefixLoops == 'add':
        full_loop[0] = 'chr' + full_loop[0].astype(str)
        full_loop[3] = 'chr' + full_loop[3].astype(str)
    elif pChrPrefixLoops == 'remove':
        full_loop[0] = full_loop[0].str.lstrip('chr')
        full_loop[3] = full_loop[3].str.lstrip('chr')

    return full_loop


def overlapLoop(pDataFrameLoop, pDataFrameProtein):
    loop_bedtool_x = BedTool.from_dataframe(pDataFrameLoop[[0, 1, 2]])
    loop_bedtool_y = BedTool.from_dataframe(pDataFrameLoop[[3, 4, 5]])

    protein_bedtool = BedTool.from_dataframe(pDataFrameProtein)
    x = loop_bedtool_x.intersect(protein_bedtool, c=True).to_dataframe()
    y = loop_bedtool_y.intersect(protein_bedtool, c=True).to_dataframe()

    mask_x = x['name'] >= 1
    mask_y = y['name'] >= 1

    selection = (mask_x) & (mask_y)

    return selection


def correlateCool(pCoolFile, pDataFrameLoop):
    cool_obj = cooler.Cooler(pCoolFile)
    number_of_peaks = cool_obj.info['nnz']
    match_list = []
    for line in pDataFrameLoop.itertuples():
        try:
            matrix_data = cool_obj.matrix(balance=False, sparse=True).fetch(str(line[1]) + ':' + str(line[2]) + '-' + str(line[3]), str(line[4]) + ':' + str(line[5]) + '-' + str(line[6]))

            if matrix_data.nnz > 0:
                match_list.append(True)
            else:
                match_list.append(False)
        except Exception:
            match_list.append(False)

    selection = pd.Series(match_list)
    log.debug(selection)

    return number_of_peaks, selection


def correlate2DText(pDataFrameProtein, pDataFrameLoop):
    loop_bedtool_x = BedTool.from_dataframe(pDataFrameLoop[[0, 1, 2, 3, 4, 5]])

    protein_bedtool = BedTool.from_dataframe(pDataFrameProtein)
    x = loop_bedtool_x.intersect(protein_bedtool, c=True).to_dataframe()

    mask_x = x['name'] >= 1

    log.debug(mask_x)
    return mask_x


def applyBinning(pDataFrame, pBinSize):
    pDataFrame_out = pDataFrame.copy()
    pDataFrame_out[1] = (pDataFrame[1] / pBinSize).astype(int) * pBinSize
    pDataFrame_out[2] = ((pDataFrame[2] / pBinSize).astype(int) + 1) * pBinSize
    pDataFrame_out.drop_duplicates()
    bedtools_data = BedTool.from_dataframe(pDataFrame_out)
    bedtools_data = bedtools_data.merge()
    bedtools_data = bedtools_data.sort()
    return bedtools_data.to_dataframe()


def writeLoopFile(pOutFileName, pLoopDataFrame):
    pLoopDataFrame.to_csv(pOutFileName, sep='\t', header=False, index=False)


def main(args=None):

    args = parse_arguments().parse_args(args)

    if args.method == 'loops':

        loop_df = readLoopFile(args.data, args.chrPrefixLoops)
        if loop_df is None:
            log.error('Empty loop file')
            return
        loop_df_bedtool = BedTool.from_dataframe(loop_df)
        loop_df = loop_df_bedtool.sort().to_dataframe(
            disable_auto_names=True, header=None)
        if args.validationType == 'bed':
            protein_df = readProtein(args.validationData, args.chrPrefixProtein)
            if protein_df is None:
                log.error('Empty protein file')
                return
            protein_df_bedtool = BedTool.from_dataframe(protein_df)
            protein_df = protein_df_bedtool.sort().to_dataframe(
                disable_auto_names=True, header=None)

            protein_df_resolution = applyBinning(protein_df, args.resolution)

            overlap_mask_df = overlapLoop(loop_df, protein_df_resolution)
            loop_df_ = loop_df[overlap_mask_df]

            number_of_proteins = len(protein_df_resolution)
        elif args.validationType == 'cool':
            number_of_proteins, overlap_mask_df = correlateCool(args.validationData, loop_df)
            loop_df_ = loop_df[overlap_mask_df]

        print('Protein peaks: {}'.format(number_of_proteins))
        print('Matched Loops: {}'.format(len(loop_df_)))
        print('Total Loops: {}'.format(len(loop_df)))

        print('Loops match protein: {}'.format(len(loop_df_) / len(loop_df)))

        if args.outFileName:
            loop_df_ = loop_df[overlap_mask_df]
            writeLoopFile(args.outFileName + '_matched_locations', loop_df_)

            with open(args.outFileName + '_statistics', 'w') as file:
                file.write(
                    '# HiCExplorer hicValidateLocations {}\n'.format(__version__))
                file.write('# Overlap of loop file {} with protein file {}\n#\n'.format(
                    args.data, args.validationData))
                file.write('Protein peaks: {}\n'.format(
                    number_of_proteins))
                file.write('Matched Loops: {}\n'.format(len(loop_df_)))
                file.write('Total Loops: {}\n'.format(len(loop_df)))
                file.write('Loops match protein: {}\n'.format(
                    len(loop_df_) / len(loop_df)))
