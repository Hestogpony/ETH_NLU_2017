import pickle
import os
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


cfg = {
    'MODELS_PATH': 'models',
    'MODEL_NAME': '',

    'MAX_TURNS': 100,
    
    'THRESHOLD': 2,
    'PAD_ID': 0,
    'UNK_ID': 1,
    'START_ID': 2,
    'EOS_ID': 3,
    'TESTSET_SIZE': 10,
    'BUCKETS': [(8,10), (100, 100)], #[(8, 10), (12, 14), (16, 19)],
    'NUM_LAYERS': 3,
    'HIDDEN_SIZE': 256,
    'BATCH_SIZE': 64,
    'LR': 0.5,
    'MAX_GRAD_NORM': 5.0,
    'NUM_SAMPLES': 128, #512

    'TEST_MAX_SIZE': 80,
    'TESTSET_SIZE': 100,

    #whether to use pretrained

}

def adapt_to_dataset(use_cornell):
    if use_cornell:
        cfg['DATA_PATH'] = 'cornell_data'
        cfg['CONVO_FILE'] = 'movie_conversations.txt'
        cfg['LINE_FILE'] = 'movie_lines.txt'
        cfg['OUTPUT_FILE'] = 'output_convo.txt'

        cfg['PROCESSED_PATH'] = 'cornell_processed'
        cfg['CPT_PATH'] = 'cornell_checkpoints'        

    else:
        # <FL> I included as comments the variables that we don't need in our reader
        # just to provide some reference
        cfg['DATA_PATH'] = 'our_data'
        cfg['CONVO_FILE'] = 'our_conversations.txt'
        cfg['LINE_FILE'] = 'Training_Shuffled_Dataset.txt'
        cfg['OUTPUT_FILE'] = 'our_output_convo.txt'

        cfg['PROCESSED_PATH'] = 'our_processed'
        cfg['CPT_PATH'] = 'our_checkpoints'

def adapt_paths_to_model():
    cfg['PROCESSED_PATH'] = os.path.join(cfg['MODELS_PATH'], cfg['MODEL_NAME'], cfg['PROCESSED_PATH'])
    cfg['CPT_PATH'] = os.path.join(cfg['MODELS_PATH'], cfg['MODEL_NAME'], cfg['CPT_PATH'])


def save_cfg(updated_config):
    config_path = os.path.join(updated_config['MODELS_PATH'], updated_config['MODEL_NAME'], "config")
    pickle.dump(updated_config , open(config_path, "wb"))
    print("Configs saved in file: %s" % (config_path))

def load_cfg(model_name):
    config_path = os.path.join(cfg['MODELS_PATH'], model_name, "config")
    loaded_cfg = pickle.load(open(config_path, "rb"))
    print("Configs loaded from %s" % (config_path))
    return loaded_cfg



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


# ENC_VOCAB = 24471
# DEC_VOCAB = 24671
# ENC_VOCAB = 1292
# DEC_VOCAB = 1320
# ENC_VOCAB = 282
# DEC_VOCAB = 237
# ENC_VOCAB = 25889
# DEC_VOCAB = 26107


# TODO transfer this 
# ENC_VOCAB = 215
# DEC_VOCAB = 218
# ENC_VOCAB = 220
# DEC_VOCAB = 222
