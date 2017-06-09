#!/bin/bash

# Setup virtual env
pip2 install virtualenv
virtualenv tf0_11
source tf0_11/bin/activate

# If the provided wheel doesn't work for you,
# find one that corresponds to your system here: https://tensorflow.blog/2016/11/13/tensorflow-v0-11-release/
# Our code is written in Python 2.7 !
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.11.0-cp27-none-linux_x86_64.whl
sudo pip2 install --upgrade $TF_BINARY_URL
pip2 install gensim

TESTFILE=$1
python chatbot.py --save_dir=models/combine_model --test=${TESTFILE}