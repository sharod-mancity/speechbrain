"""
 -----------------------------------------------------------------------------
 test.py (author: Mirco Ravanelli)

 Description: This is the a script that performs some basic tests on the
              code of the SpeechBrain toolkit. Please, run this code before
              asking for a pull request.

 Input:       None

 Output:      None

 Example:     python test.py
 -----------------------------------------------------------------------------
"""

import sys
import pycodestyle
from speechbrain.utils import create_exec_config
from speechbrain.core import execute_computations
from glob import glob

# List of config files to run:
cfg_lst = [
    "cfg/minimal_examples/data_reading/read_write_data.cfg",
    "cfg/minimal_examples/data_reading/loop_example.cfg",
    "cfg/minimal_examples/basic_processing/"
    "minimal_processing_read_write_example_noise.cfg",
    "cfg/minimal_examples/features/compute_stft_example.cfg",
    "cfg/minimal_examples/features/" "compute_spectrogram_example.cfg",
    "cfg/minimal_examples/features/compute_fbanks_example.cfg",
    "cfg/minimal_examples/features/compute_mfccs_example.cfg",
    "cfg/minimal_examples/features/compute_mfccs_example2.cfg",
    "cfg/minimal_examples/multichannel/"
    "compute_stft_multichannel_example.cfg",
    "cfg/minimal_examples/multichannel/"
    "compute_spectrogram_multichannel_example.cfg",
    "cfg/minimal_examples/multichannel/"
    "compute_fbanks_multichannel_example.cfg",
    "cfg/minimal_examples/neural_networks/spk_id/spk_id_example.cfg",
    "cfg/minimal_examples/neural_networks/DNN_HMM_ASR/ASR_example.cfg",
    "cfg/minimal_examples/neural_networks/autoencoder/autoencoder_example.cfg",
    "cfg/minimal_examples/neural_networks/E2E_ASR/CTC/CTC_example.cfg",
]

augmentation_config_list = glob(
    'cfg/minimal_examples/basic_processing/minimal*.cfg'
)

cfg_lst += augmentation_config_list

# List of files to check:
check_lst = [
'spbrain.py',
'speechbrain/core.py',
'speechbrain/data_io/data_io.py',
'speechbrain/data_io/data_preparation.py',
'speechbrain/decoders/decoders.py',
'speechbrain/processing/features.py',
'speechbrain/processing/multi_mic.py',
'speechbrain/processing/speech_augmentation.py',
'speechbrain/utils/input_validation.py',
'speechbrain/utils/data_utils.py',
'speechbrain/utils/config.py',
'speechbrain/utils/edit_distance.py',
'speechbrain/utils/superpowers.py',
'speechbrain/utils/logger.py',
'speechbrain/utils/__init__.py',
'speechbrain/nnet/normalization.py',
'speechbrain/nnet/losses.py',
'speechbrain/nnet/lr_scheduling.py',
'speechbrain/nnet/architectures.py',
'speechbrain/nnet/optimizers.py'
]

# Running examples in config files
for cfg_file in cfg_lst:

    print("checking %s" % cfg_file)
    # Creating config dict for executing computations
    exec_config = create_exec_config(cfg_file, "")

    # Initializing the execute computation class
    computations = execute_computations(exec_config)

    # Executing the computations specified in the config file
    computations([])

# Checking PEP8 consistency
print("check PEP8 consistency:")

for file in check_lst:

    # Checking if the file is compliant with PEP8
    fchecker = pycodestyle.Checker(file, show_source=True)
    file_errors = fchecker.check_all()

    print("Found %s errors (and warnings) in %s" % (file_errors, file))

    if file_errors > 0:
        sys.exit(0)

print("Test finished without errors!")