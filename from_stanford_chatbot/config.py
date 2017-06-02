""" A neural chatbot using sequence to sequence model with
attentional decoder.

This is based on Google Translate Tensorflow model
https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/

Sequence to sequence model by Cho et al.(2014)

Created by Chip Huyen as the starter code for assignment 3,
class CS 20SI: "TensorFlow for Deep Learning Research"
cs20si.stanford.edu

This file contains the hyperparameters for the model.

See readme.md for instruction on how to run the starter code.
"""

# parameters for processing the dataset

USE_CORNELL = False

if USE_CORNELL:
    DATA_PATH = 'cornell_data'
    CONVO_FILE = 'movie_conversations.txt'
    LINE_FILE = 'movie_lines.txt'
    OUTPUT_FILE = 'output_convo.txt'
    PROCESSED_PATH = 'cornell_processed'
    CPT_PATH = 'cornell_checkpoints'
else:
    # <FL> I included as comments the variables that we don't need in our reader
    # just to provide some reference
    DATA_PATH = 'our_data'
    # CONVO_FILE = None
    LINE_FILE = 'Training_Shuffled_Dataset.txt'
    # OUTPUT_FILE = None
    PROCESSED_PATH = 'our_processed'
    CPT_PATH = 'our_checkpoints'

    MAX_TURNS = 1000

# <FL> When do we cut off a sentence
TEST_MAX_SIZE = 80

THRESHOLD = 2

PAD_ID = 0
UNK_ID = 1
START_ID = 2
EOS_ID = 3

TESTSET_SIZE = 100

# model parameters
""" Train encoder length distribution:
[175, 92, 11883, 8387, 10656, 13613, 13480, 12850, 11802, 10165,
8973, 7731, 7005, 6073, 5521, 5020, 4530, 4421, 3746, 3474, 3192,
2724, 2587, 2413, 2252, 2015, 1816, 1728, 1555, 1392, 1327, 1248,
1128, 1084, 1010, 884, 843, 755, 705, 660, 649, 594, 558, 517, 475,
426, 444, 388, 349, 337]
These buckets size seem to work the best
"""
# [19530, 17449, 17585, 23444, 22884, 16435, 17085, 18291, 18931]
# BUCKETS = [(6, 8), (8, 10), (10, 12), (13, 15), (16, 19), (19, 22), (23, 26), (29, 32), (39, 44)]

# [37049, 33519, 30223, 33513, 37371]
# BUCKETS = [(8, 10), (12, 14), (16, 19), (23, 26), (39, 43)]

BUCKETS = [(8, 10), (12, 14), (16, 19), (40, 40), (80, 80)]

NUM_LAYERS = 3
HIDDEN_SIZE = 256
BATCH_SIZE = 64

LR = 0.5
MAX_GRAD_NORM = 5.0

NUM_SAMPLES = 512
ENC_VOCAB = 24471
DEC_VOCAB = 24671
ENC_VOCAB = 1292
DEC_VOCAB = 1320
ENC_VOCAB = 1285
DEC_VOCAB = 1313
