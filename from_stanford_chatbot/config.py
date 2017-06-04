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

# <BG> Original parameters are are copied to comments in case of changes 

cfg = {
    'MODELS_PATH': 'models',
    'MODEL_NAME': '', # leave this empty, this is filled with a timestamp

    'MAX_TURNS': -1, # Number of conversations to read in
    'SKIP_STEP': 500, #  # After that many batches, there's a validation run.

    'TESTSET_SIZE': 100, # 25000    # Size of the evaluation set
    'TEST_MAX_LENGTH': 80, # Maximum length of sentences in the test set
    
    'THRESHOLD': 2, #2      # A word has to appear this many times to be part of the vocabulary

    'PAD_ID': 0,
    'UNK_ID': 1,
    'START_ID': 2,
    'EOS_ID': 3,

    'BUCKETS': [(30,30)],
    # original 'BUCKETS': [(8, 10), (12, 14), (16, 19), (23, 26), (39, 43)],
    # small version 'BUCKETS': [(8, 10), (12, 14), (16, 19)], 
    #our numbers 'BUCKETS': [(6, 8), (8, 10), (10, 12), (13, 15), (16, 19), (19, 22), (23, 26), (29, 32), (39, 44),(50,50),(60,60)],#[(8,10), (16, 19)], #[(8, 10), (12, 14), (16, 19)],
    'NUM_LAYERS': 1, #3     # Recurrent Layers in the Mulit-Layer-RNN
    'HIDDEN_SIZE': 256,
    'BATCH_SIZE': 64,
    'LR': 0.5,
    'MAX_GRAD_NORM': 5.0,
    'NUM_SAMPLES': 3, #512      # for sampled softmax loss

    'STANDARD_SOFTMAX': False,
    'KEEP_PREV': False,
    'EPOCHS': 100,
    'DROPOUT_RATE': 0.3,

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
    # print("Configs saved in file: %s" % (config_path))

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


# <BG> original vocab sizes on the Cornell data set
# ENC_VOCAB = 24471
# DEC_VOCAB = 24671

