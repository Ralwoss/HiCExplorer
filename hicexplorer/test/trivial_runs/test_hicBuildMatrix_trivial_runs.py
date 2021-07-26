"""
    Testsuite for testing if all components of hiCexplorer are working without raising
    Exceptions.

    This means all tests are designed to fail if there occur any Exceptions in trivial
    calls of the code using the argparser`s arguments.
"""
import warnings
warnings.simplefilter(action="ignore", category=RuntimeWarning)
warnings.simplefilter(action="ignore", category=PendingDeprecationWarning)
import pytest
from tempfile import NamedTemporaryFile, mkdtemp
import os
import shutil
from psutil import virtual_memory

from hicexplorer.utilities import genomicRegion
from hicexplorer import hicBuildMatrix as hicBuildMatrix
from hicexplorer.test.test_compute_function import compute

mem = virtual_memory()
memory = mem.total / 2**30

# memory in GB the test computer needs to have to run the test case
LOW_MEMORY = 2
MID_MEMORY = 4
HIGH_MEMORY = 120

REMOVE_OUTPUT = True


# Some definitions needed for tests
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_data/")
# test_build_matrix
sam_R1 = ROOT + "small_test_R1_unsorted.bam"
sam_R2 = ROOT + "small_test_R2_unsorted.bam"
bam_R1 = ROOT + "R1_1000.bam"
bam_R2 = ROOT + "R2_1000.bam"
dpnii_file = ROOT + "DpnII.bed"
outFile = NamedTemporaryFile(suffix='.h5', delete=False)
qc_folder = mkdtemp(prefix="testQC_")


@pytest.mark.parametrize("sam1", [bam_R1])  # required
@pytest.mark.parametrize("sam2", [bam_R2])  # required
@pytest.mark.parametrize("outFile", [outFile])  # required
@pytest.mark.parametrize("qcFolder", [qc_folder])  # required
@pytest.mark.parametrize("outBam", ['/tmp/test.bam'])
@pytest.mark.parametrize("binSize", [100000])  # required | restrictionCutFile
@pytest.mark.parametrize("restrictionCutFile", [dpnii_file])  # required | binSize
@pytest.mark.parametrize("minDistance", [150])
@pytest.mark.parametrize("maxDistance", [1500])
@pytest.mark.parametrize("maxLibraryInsertSize", [1500])
@pytest.mark.parametrize("restrictionSequence", ['GATC'])
@pytest.mark.parametrize("danglingSequence", ['GATC'])
@pytest.mark.parametrize("region", ["ChrX"])  # region does not work!!
@pytest.mark.parametrize("minMappingQuality", [15])
@pytest.mark.parametrize("threads", [4])
@pytest.mark.parametrize("inputBufferSize", [400000])
def test_build_matrix_restrictionCutFile_two(sam1, sam2, outFile, qcFolder, outBam, binSize,
                                             restrictionCutFile, minDistance, maxDistance,
                                             maxLibraryInsertSize, restrictionSequence,
                                             danglingSequence, region, 
                                             minMappingQuality, threads, inputBufferSize):
    # test more args for restrictionCutFile option
    region = genomicRegion(region)

    args = "-s {} {} --restrictionCutFile {} --outFileName {} --QCfolder {} " \
           "--restrictionSequence {} " \
           "--danglingSequence {} " \
           "--minDistance {} " \
           "--maxLibraryInsertSize {} --threads {} " \
           "--region {}  ".format(bam_R1, bam_R2,
                                                         restrictionCutFile, outFile.name,
                                                         qcFolder,
                                                         restrictionSequence,
                                                         danglingSequence,
                                                         minDistance,
                                                         maxLibraryInsertSize,
                                                         threads, region
                                                         ).split()
    # hicBuildMatrix.main(args)
    compute(hicBuildMatrix.main, args, 5)

    os.unlink(outFile.name)
    shutil.rmtree(qcFolder)
    # os.unlink("/tmp/test.bam")


@pytest.mark.parametrize("sam1", [bam_R1])  # required
@pytest.mark.parametrize("sam2", [bam_R2])  # required
@pytest.mark.parametrize("outFile", [outFile])  # required
@pytest.mark.parametrize("qcFolder", [qc_folder])  # required
@pytest.mark.parametrize("outBam", ['/tmp/test.bam'])
@pytest.mark.parametrize("binSize", [100000])  # required | restrictionCutFile
@pytest.mark.parametrize("restrictionCutFile", [dpnii_file])  # required | binSize
@pytest.mark.parametrize("minDistance", [150])
@pytest.mark.parametrize("maxDistance", [1500])
@pytest.mark.parametrize("maxLibraryInsertSize", [1500])
@pytest.mark.parametrize("restrictionSequence", ['GATC'])
@pytest.mark.parametrize("danglingSequence", ['GATC'])
@pytest.mark.parametrize("region", ["ChrX"])  # region does not work!!
@pytest.mark.parametrize("minMappingQuality", [15])
@pytest.mark.parametrize("threads", [4])
@pytest.mark.parametrize("inputBufferSize", [400000])
def test_build_matrix_restrictionCutFile_three(sam1, sam2, outFile, qcFolder, outBam, binSize,
                                               restrictionCutFile, minDistance, maxDistance,
                                               maxLibraryInsertSize, restrictionSequence,
                                               danglingSequence, region,
                                               minMappingQuality, threads, inputBufferSize):
    # test more params with restrictionCutFile (now without region param)
    args = "-s {} {} --restrictionCutFile {} --outFileName {} --QCfolder {} " \
           "--restrictionSequence {} " \
           "--danglingSequence {} " \
           "--minDistance {} " \
           "--maxLibraryInsertSize {} --threads {} " \
           " ".format(bam_R1, bam_R2,
                                             restrictionCutFile, outFile.name, qcFolder,
                                             restrictionSequence, danglingSequence,
                                             minDistance, maxLibraryInsertSize, threads
                                             ).split()

    # hicBuildMatrix.main(args)
    compute(hicBuildMatrix.main, args, 5)

    os.unlink(outFile.name)
    shutil.rmtree(qcFolder)
    # os.unlink("/tmp/test.bam")


@pytest.mark.parametrize("sam1", [bam_R1])  # required
@pytest.mark.parametrize("sam2", [bam_R2])  # required
@pytest.mark.parametrize("outFile", [outFile])  # required
@pytest.mark.parametrize("qcFolder", [qc_folder])  # required
@pytest.mark.parametrize("outBam", ['/tmp/test.bam'])
@pytest.mark.parametrize("binSize", [100000])  # required | restrictionCutFile
@pytest.mark.parametrize("restrictionCutFile", [dpnii_file])  # required | binSize
@pytest.mark.parametrize("minDistance", [150])
@pytest.mark.parametrize("maxDistance", [1500])
@pytest.mark.parametrize("maxLibraryInsertSize", [1500])
@pytest.mark.parametrize("restrictionSequence", ['GATC'])
@pytest.mark.parametrize("danglingSequence", ['GATC'])
@pytest.mark.parametrize("region", ["ChrX"])  # region does not work!!
@pytest.mark.parametrize("minMappingQuality", [15])
@pytest.mark.parametrize("threads", [4])
@pytest.mark.parametrize("inputBufferSize", [400000])
def test_build_matrix_restrictionCutFile_four(sam1, sam2, outFile, qcFolder, outBam, binSize,
                                              restrictionCutFile, minDistance, maxDistance,
                                              maxLibraryInsertSize, restrictionSequence,
                                              danglingSequence, region,
                                              minMappingQuality, threads, inputBufferSize):
    # test more params with restrictionCutFile (now without region param)
    args = "-s {} {} --restrictionCutFile {} --outFileName {} --QCfolder {} " \
           "--restrictionSequence {} " \
           "--danglingSequence {} " \
           "--minDistance {} " \
           "--maxLibraryInsertSize {} --threads {} " \
           " --keepSelfCircles ".format(bam_R1, bam_R2,
                                                               restrictionCutFile,
                                                               outFile.name, qcFolder,
                                                               restrictionSequence,
                                                               danglingSequence,
                                                               minDistance,
                                                               maxLibraryInsertSize,
                                                               threads
                                                               ).split()

    # hicBuildMatrix.main(args)
    compute(hicBuildMatrix.main, args, 5)

    os.unlink(outFile.name)
    shutil.rmtree(qcFolder)
    # os.unlink("/tmp/test.bam")
