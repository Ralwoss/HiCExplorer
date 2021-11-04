import warnings
import argparse
import cooler as cool
import numpy as np
import pyBigWig
from tensorflow.keras.utils import Sequence
import tensorflow as tf
import random

from hicexplorer.parserCommon import CustomFormatter
from hicexplorer._version import __version__


import logging
log = logging.getLogger(__name__)

warnings.simplefilter(action="ignore", category=RuntimeWarning)
warnings.simplefilter(action="ignore", category=PendingDeprecationWarning)




# Try to load submatrices when needed in dataGenerator. Takes too long
class dataGenerator(Sequence):
    def __init__(self, cooler, windows, window_values, chromosomes, window_size, center_radius, batch_size=32,
                 n_classes=2, shuffle=True):

        self.cooler = cooler
        self.matrix = cooler.matrix(balance = False) #do not try balance = True, does not work
        self.windows = windows
        self.chromosomes = chromosomes
        self.dim = (window_size**2,)
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.center_radius = center_radius
        self.X_pos, self.X_neg = self.prepare_Xs(windows, window_values)

        self.on_epoch_end()

    def on_epoch_end(self):
            # for smaller sample set multiply size to get more indexes for sampling
        maxsize = max(len(self.X_pos), len(self.X_neg))
        if len(self.X_pos) < maxsize:  # if less positive samples
            proportion = int(np.ceil(maxsize / len(self.X_pos)))
            self.indexes_pos = np.arange(len(self.X_pos) * proportion)
            self.indexes_neg = np.arange(len(self.X_neg))
        elif len(self.X_neg) < maxsize:  # if less negative samples
            proportion = int(np.ceil(maxsize / len(self.X_neg)))
            self.indexes_pos = np.arange(len(self.X_pos))
            self.indexes_neg = np.arange(len(self.X_neg) * proportion)
        else:  # if equal positive and negative samples
            self.indexes_pos = np.arange(len(self.X_pos))
            self.indexes_neg = np.arange(len(self.X_neg))

        # print(self.indexes)
        if self.shuffle:
            np.random.shuffle(self.indexes_pos)
            np.random.shuffle(self.indexes_neg)


    def __getitem__(self, index):
        # generate indexes for this batch
        # division by two to get half the number of indexes for pos and neg index lists
        indexes_pos = self.indexes_pos[int((index * self.batch_size) / 2):int(((index + 1) * self.batch_size) / 2)]
        indexes_neg = self.indexes_neg[int((index * self.batch_size) / 2):int(((index + 1) * self.batch_size) / 2)]
        X, y = self.__data_generation(indexes_pos, indexes_neg)

        return X, y

    def __len__(self):
        return int(np.floor(2 * max(len(self.X_pos), len(self.X_neg)) / self.batch_size))

    def __data_generation(self, indexes_pos, indexes_neg):
        """Generates data containing batch_size samples"""  # X : (n_samples, *dim, n_channels)
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty(self.batch_size, dtype=int)

        for i in np.arange(len(indexes_pos)):
            X[2 * i,] = self.X_pos[indexes_pos[i] % len(self.X_pos)]
            y[2 * i] = 1
            X[2 * i + 1] = self.X_neg[indexes_neg[i] % len(self.X_neg)]
            y[2 * i + 1] = 0

        return X, y

    def prepare_Xs(self, windows, window_values):
        X_pos = []
        X_neg = []
        for chrom in windows:
            for index in range(len(windows[chrom])):
                bounds = windows[chrom][index]
                value = window_values[chrom][index]
                wstart = bounds[0]
                wend = bounds[1]

                random_addition = random.randint(-self.center_radius, self.center_radius)
                if value:
                    if not (wstart + random_addition < 0 or wend + random_addition >= len(self.matrix)):
                        wstart, wend = wstart + random_addition, wend + random_addition
                    sm = (self.matrix[wstart:wend, wstart:wend]).flatten()
                    X_pos.append(sm)
                else:
                    sm = (self.matrix[wstart:wend, wstart:wend]).flatten()
                    X_neg.append(sm)
        return X_pos, X_neg


def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=CustomFormatter,
        add_help=False,
        conflict_handler='resolve',
        description="""
Trains a neural network for predicting TAD boundaries on BED-file with TAD boundaries and corresponding Hi-C matrix

    $ hicBuildNeuralNetwork --matrix hic_matrix.cool --boundaries boundaries.bed --binsize 10000 --modelOut path/to/model

"""
    )

    parserRequired = parser.add_argument_group('Required arguments')

    parserRequired.add_argument('--matrix', '-m',
                                help='HiCExplorer matrix in cooler format.',
                                required=True)

    parserRequired.add_argument('--boundaries', '-b',
                                help='TAD boundaries in Form of bed file or tab separated csv '
                                     'in the form of \'chr start end\'',
                                required=True)

    parserRequired.add_argument('--binsize',
                                help='bin size of the Hi-C matrix',
                                type=int,
                                required=True)

    parserRequired.add_argument('--modelOut', '-o',
                                help='Path under which the trained model will be saved.',
                                required=True)





    parserOpt = parser.add_argument_group('Optional arguments')

    parserOpt.add_argument('--modelInfo',
                           help='If True a txt is written with information about the model'
                                'evaluated on the validation and test data. '
                                'If no validation and test data is given, there is no effect',
                           choices=[None, "validation", "test"],
                           default=None,
                           required=False)

    parserOpt.add_argument('--validationChroms',
                           help='chromosomes you wish to add to the validation set.'
                           'Chromosomes not added to validation or test sets will be added to training set',
                           nargs='+',
                           required=False)
    parserOpt.add_argument('--testChroms',
                           help='chromosomes you wish to add to the test set.'
                                'Chromosomes not added to validation or test sets will be added to training set',
                           nargs='+',
                           required=False)

    parserOpt.add_argument('--windowRadius',
                           help='Radius of window to construct windows for reading data from HiC matrix. '
                                'Window size = 2 * windowRadius + 1',
                           type=int,
                           default = '7',
                           required=False)
    parserOpt.add_argument('--centerRadius',
                           help='Radius of center of window in which TADs are recognized.'
                                'Window is shifted to the right by \'Center size\''
                                'Center size = 2 * centerRadius + 1. ',
                           type=int,
                           default= '2',
                           required=False)
    parserOpt.add_argument('--indexShiftValue',
                           help='Number of bins that the window will be shifted to the right.'
                                'if no value is given, window will shift the size of the center.',
                           type=int,
                           default=None,
                           required=False)

    parserOpt.add_argument('--aCompartment',
                           help="Should only the A compartment be used when training the network",
                           choices=["True", "False"],
                           default=False,
                           required= False)

    parserOpt.add_argument('--aCompartmentEigenvectors',
                           help="All eigenvector Files in bigwig format that are used to find A compartments."
                                "If aCompartment is False it has no impact",
                           nargs='+',
                           default=None,
                           required=False)
    parserOpt.add_argument('--aCompartmentInformation',
                           help="A Textfile for identifying the right compartments as A compartments"
                                " from the eigenvectors is needed. The Textfile has to have a tab-separated table"
                                " in the form of: "
                                "Chromosome[tab]eigenvectorName[tab]sign "
                                "Chromosome must be the exact name from the Hi-C matrix. "
                                "The eigenvectorName must be the exact filename given in aCompartmentEigenvectors. "
                                "The eigenvectorName identifies the eigenvector that corresponds"
                                " to the compartmentalization. "
                                "The sign must be either '+' or '-'. "
                                "The sign informs if positive or negative eigenvalues correspond to an A compartment."
                                "If aCompartment is False it has no impact",
                           default=None,
                           required=False)

    parserOpt.add_argument('--help', '-h', action='help', help='show the help '
                           'message and exit')

    parserOpt.add_argument('--version', action='version',
                           version='%(prog)s {}'.format(__version__))

    return parser


def aCompartment(boundaries, binsize, eigenvectors, informationFile):
    chromosome_infos = {}
    with open(informationFile, "r") as f:
        for line in f:
            cont = line.strip().split()
            chromosome_infos[cont[0]] = [cont[1],cont[2]]
    for chrom, info in chromosome_infos.items():
        eigenvector = None
        for ev in eigenvectors:
            if ev.endswith(info[0]):
                eigenvector = pyBigWig.open(ev, "r")
        if eigenvector == None:
            log.error(f"The given eigenvector of Chromosome {chrom} does not exist.")
            exit(1)
        if info[1] == "+":
            sign = 1
        elif info[1] == "-":
            sign = -1
        else:
            log.error(f"No valid sign for Chromosome {chrom} is given.")
        new_bounds = []
        for bound in boundaries[chrom]:
            if bound * binsize >eigenvector.chroms(chrom):
                continue
            if sign * eigenvector.stats(chrom, int(bound * binsize),
                                        min(int((bound + 1) * binsize), eigenvector.chroms(chrom)))[0] >= 0:
                # if sign * eigenvector of boundary bin under 0 delete boundary
                new_bounds.append(bound)

        boundaries[chrom] = new_bounds

    return boundaries


def make_val_data(hic_cooler, boundaries, validation_chroms, window_radius, center_radius, shift_value):
    print("Constructing validation set:")
    windows = {} # dict with key:chrom, value:[startindex, endindex]
    window_values = {} # dict with key:chrom value:window_value
    for chrom in hic_cooler.chromnames:
        if chrom not in validation_chroms or chrom not in boundaries.keys():
            continue
        else:
            windows[chrom] = []
            window_values[chrom] = []
        print(f"\tConstructing windows for Chromosome {chrom}")
        chrbounds = boundaries[chrom]
        offset = hic_cooler.offset(chrom)
        wstart = offset
        wend = wstart + 2 * window_radius + 1
        boundary_index = 0
        while wend <= hic_cooler.extent(chrom)[1]:
            while chrbounds[boundary_index] < wstart - offset:
                if boundary_index >= len(chrbounds) - 1:
                    break
                boundary_index += 1
            label = 0
            windows[chrom].append([wstart, wend])
            for bound in boundaries[chrom][boundary_index:]:
                if bound > wend - offset:
                    break
                if wstart - offset + window_radius - center_radius <= bound \
                        < wend - offset - window_radius + center_radius:
                    label = 1

                if label == 1:
                    window_values[chrom].append(1)
                    break
            if label == 0:
                window_values[chrom].append(0)
            if shift_value is not None:
                wstart, wend = wstart + int(shift_value), wend + int(shift_value)
            else:
                wstart, wend = wstart + 2 * center_radius + 1, wend + 2 * center_radius + 1
    return windows, window_values

def make_test_data(hic_cooler, boundaries, test_chroms, window_radius, center_radius, shift_value):
    print("Constructing test set:")
    windows = {} # dict with key:chrom, value:[startindex, endindex]
    window_values = {} # dict with key:chrom value:window_value
    for chrom in hic_cooler.chromnames:
        if chrom not in test_chroms or chrom not in boundaries.keys():
            continue
        else:
            windows[chrom] = []
            window_values[chrom] = []
        print(f"\tConstructing windows for Chromosome {chrom}")
        chrbounds = boundaries[chrom]
        offset = hic_cooler.offset(chrom)
        wstart = offset
        wend = wstart + 2 * window_radius + 1
        boundary_index = 0
        while wend <= hic_cooler.extent(chrom)[1]:
            while chrbounds[boundary_index] < wstart - offset:
                if boundary_index >= len(chrbounds) - 1:
                    break
                boundary_index += 1
            label = 0
            windows[chrom].append([wstart, wend])
            for bound in boundaries[chrom][boundary_index:]:
                if bound > wend - offset:
                    break
                if wstart - offset + window_radius - center_radius <= bound \
                        < wend - offset - window_radius + center_radius:
                    label = 1

                if label == 1:
                    window_values[chrom].append(1)
                    break
            if label == 0:
                window_values[chrom].append(0)
            if shift_value is not None:
                wstart, wend = wstart + int(shift_value), wend + int(shift_value)
            else:
                wstart, wend = wstart + 2 * center_radius + 1, wend + 2 * center_radius + 1
    return windows, window_values


def make_train_data(hic_cooler, boundaries, train_chroms, window_radius, center_radius, shift_value):
    print("Constructing training set:")
    windows = {}  # dict with key:chrom, value:[startindex, endindex]
    window_values = {}  # dict with key:chrom value:window_value
    for chrom in hic_cooler.chromnames:
        if chrom not in train_chroms or chrom not in boundaries.keys():
            continue
        else:
            windows[chrom] = []
            window_values[chrom] = []
        print(f"\tConstructing windows for Chromosome {chrom}")
        chrbounds = boundaries[chrom]
        offset = hic_cooler.offset(chrom)
        for bound in chrbounds:
            if (bound < window_radius - center_radius):
                continue
            wstart = int(bound - window_radius) + offset
            wend = int(bound + window_radius) + offset + 1
            windows[chrom].append([wstart, wend])
            window_values[chrom].append(1)

        wstart = offset
        wend = wstart + 2 * window_radius + 1
        boundary_index = 0
        while wend <= hic_cooler.extent(chrom)[1]:
            while chrbounds[boundary_index] < wstart - offset:
                if boundary_index >= len(chrbounds) - 1:
                    break
                boundary_index += 1
            label = 0

            for bound in boundaries[chrom][boundary_index:]:
                if bound > wend - offset:
                    break
                if wstart - offset + window_radius - center_radius <= bound \
                        < wend - offset - window_radius + center_radius:
                    label = 1

                if label == 1:
                    break
            if label == 0:
                windows[chrom].append([wstart, wend])
                window_values[chrom].append(0)
            if shift_value is not None:
                wstart, wend = wstart + int(shift_value), wend + int(shift_value)
            else:
                wstart, wend = wstart + 2 * center_radius + 1, wend + 2 * center_radius + 1
    return windows, window_values

def construct_neural_network(window_size):

    METRICS = [

        tf.keras.metrics.TruePositives(name='tp'),

        tf.keras.metrics.FalsePositives(name='fp'),

        tf.keras.metrics.TrueNegatives(name='tn'),

        tf.keras.metrics.FalseNegatives(name='fn'),

        tf.keras.metrics.BinaryAccuracy(name='accuracy'),

        tf.keras.metrics.Precision(name='precision')
    ]

    """
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(window_size ** 2,)))
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.2)),
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    """

    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(window_size ** 2,)))
    model.add(tf.keras.layers.Dense(96, activation="selu"))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(64,"elu"))
    model.add(tf.keras.layers.Dense(192, "selu"))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(1,activation="sigmoid"))


    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0027667399983716005),
                  loss='binary_crossentropy',
                  metrics=METRICS)


    return model


def train_neural_network(hic_cooler, train_data, train_chroms ,window_size, center_radius):
    X_train = train_data[0]
    y_train = train_data[1]
    model = construct_neural_network(window_size)
    #cooler, windows, window_values, chromosomes, window_size, binsize, center_radius,
    dg = dataGenerator(hic_cooler, X_train, y_train ,train_chroms, window_size, center_radius)

    model.fit(dg, epochs=100)

    return model

def evaluate_model(hic_cooler, window_size, center_size, step_size, model, data, model_path):
    with open(model_path + ".info", "w")as f:
        m = hic_cooler.matrix(balance=False)
        f.write("general details:\n")
        f.write(f"\twindow_size: {window_size}\n")
        f.write(f"\tcenter_size: {center_size}\n")
        if step_size is None:
            f.write(f"\tindex_shift_value: {center_size}\n")
        else:
            f.write(f"\tindex_shift_value: {step_size}\n")
        f.write("\n")
        f.write("\n")
        f.write("\n")


        if len(data[0]) > 0:
            X=data[0]
            y=data[1]

            true_pos = 0
            true_neg = 0
            false_pos = 0
            false_neg = 0
            total = 0

            chrom_string = "\tChromosomes:"
            Xnew = []
            ynew = []
            for ch in X.keys():
                sms = []
                for bounds in X[ch]:
                    sms.append(np.array(m[bounds[0]: bounds[1], bounds[0]:bounds[1]]).flatten())

                Xnew = Xnew + sms
                sms = np.array(sms)
                ynew = ynew + data[1][ch]
                yval_pred = model.predict(sms)
                total += len(yval_pred)
                chrom_string = f"{chrom_string}{ch}, "
                for i in range(len(yval_pred)):
                    if abs(y[ch][i] - yval_pred[i]) < 0.5:
                        if y[ch][i] == 1:
                            true_pos += 1
                        else:
                            true_neg += 1
                    else:
                        if y[ch][i] == 1:
                            false_neg += 1
                        else:
                            false_pos += 1
            f.write("Results of the data:\n")
            f.write(f"{chrom_string[:-2]}\n")
            f.write(f"\ttotal predictions: {total}\n")
            f.write(f"\ttrue positives: {true_pos}\n")
            f.write(f"\ttrue negatives: {true_neg}\n")
            f.write(f"\tfalse positives: {false_pos}\n")
            f.write(f"\tfalse negatives: {false_neg}\n")
            eval = model.evaluate(np.array(Xnew), np.array(ynew))
            f.write(f"\tevaluation: {eval}")
            f.write("\n")
            f.write("\n")
            f.write("\n")


def main(args=None):
    args = parse_arguments().parse_args(args)
    if args.windowRadius < args.centerRadius:
        log.error("CenterRadius can not be larger than window radius.")
        exit(1)
    c = cool.Cooler(args.matrix)
    m = c.matrix(balance=False)
    binsize = args.binsize

    boundaries = {} # map for saving list of boundaries per chromosome

    with open(args.boundaries, "r") as f:
        for line in f:
            cont = line.strip().split()
            if cont[0] not in boundaries: # make new entry for chrom in dict if chrom not in dict
                boundaries[cont[0]] = [int(cont[1]) / binsize]  # Add first boundary
            boundaries[cont[0]].append((int(cont[2]) / binsize))  # append every end boundary of TAD
    a_compartment = args.aCompartment
    if a_compartment == "True":
        aCompartment(boundaries, binsize, args.aCompartmentEigenvectors, args.aCompartmentInformation)
    train_chroms = c.chromnames
    for chrom in args.validationChroms: train_chroms.remove(chrom)
    for chrom in args.testChroms: train_chroms.remove(chrom)

    val_data = make_val_data(c, boundaries, args.validationChroms, args.windowRadius,
                             args.centerRadius, args.indexShiftValue)
    test_data = make_test_data(c, boundaries, args.testChroms, args.windowRadius,
                               args.centerRadius, args.indexShiftValue)
    train_data = make_train_data(c, boundaries, train_chroms, args.windowRadius,
                                 args.centerRadius, args.indexShiftValue)

    model = train_neural_network(c, train_data, train_chroms, 2*args.windowRadius+1, args.centerRadius)

    model.save(args.modelOut)


    if (len(val_data[0]) > 0 or len(test_data[0]) > 0) and args.modelInfo == "validation":
        evaluate_model(c, args.windowRadius * 2 + 1,args.centerRadius * 2 + 1, args.indexShiftValue,
                       model, val_data, args.modelOut)
    elif (len(val_data[0]) > 0 or len(test_data[0]) > 0) and args.modelInfo == "test":
        evaluate_model(c, args.windowRadius * 2 + 1,args.centerRadius * 2 + 1, args.indexShiftValue,
                       model, test_data, args.modelOut)

    return





