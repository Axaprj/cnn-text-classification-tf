#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
import flags
import subprocess
from text_cnn import TextCNN
from tensorflow.contrib import learn

# Parameters
# ==================================================
FLAGS = tf.flags.FLAGS


def preprocess():
    # Data Preparation
    # ==================================================

    # Load data
    print("Loading data...")
    x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file,
                                                  FLAGS.negative_data_file)
    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text])

    dir_path = os.path.dirname(os.path.realpath(__file__)) + "/fasttext/"
    print("0: " + dir_path)
    p = subprocess.Popen([
        dir_path + "fasttext", "print-word-vectors", dir_path + "cc.en.300.bin"
    ],
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)
    print("1")
    x_list=[]
    for xt in x_text:
        ws, vs = GetFasttextArr(p, xt)
        z_embed = np.zeros([max_document_length - vs.shape[0], 300])
        vs =  np.vstack((vs, z_embed))
        x_list.append(vs)
    #ws, vs = GetFasttextArr(p, "Martin Luther")
    #z_embed = np.zeros([max_document_length - vs.shape[0], 300])
    #vs =  np.vstack((vs, z_embed))
    #x_list.append(vs)
    print("Done.")
    return np.stack(x_list), y

def GetFasttextArr(proc_ft, in_str):
    proc_ft.stdin.write((in_str + ' <%EOL%>\n').encode('utf-8'))
    proc_ft.stdin.flush()
    is_eol = False
    word_list = []
    vect_list = np.empty([0, 300])
    while not is_eol:
        l = proc_ft.stdout.readline().decode("utf-8")
        l = l.strip(' \t\n\r')
        is_eol = l.startswith('<%EOL%>')
        if not is_eol:
            lsplit = l.split(" ")
            word_list.append(lsplit[0])
            snp = np.array(lsplit[1:])
            fnp = snp.astype(np.float)
            vect_list = np.vstack((vect_list, fnp))
    print(word_list)
    return word_list, vect_list


def main(argv=None):
    x, y = preprocess()
    print(x.size, y.size)
    np.savez(FLAGS.data_file, x=x, y=y)


if __name__ == '__main__':
    tf.app.run()