#!/bin/bash

# Execute in su mode

# Setup virtual env
# Make sure you use python2
pip install virtualenv
echo 'pip2 install virtualenv'
virtualenv tf0_11
echo 'virtualenv tf0_11'
. ./tf0_11/bin/activate
echo 'tf0_11/bin/activate'
echo ${VIRTUAL_ENV}

# If the provided wheel doesn't work for you,
# find one that corresponds to your system here: https://tensorflow.blog/2016/11/13/tensorflow-v0-11-release/
# Our code is written in Python 2.7 !
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.11.0-cp27-none-linux_x86_64.whl
pip install --upgrade $TF_BINARY_URL
pip install gensim

TESTFILE=$1
python chatbot.py --save_dir=models/combined_model --relevance=0.3 --beam_width=2 --temperature=1.0 --test=${TESTFILE}
