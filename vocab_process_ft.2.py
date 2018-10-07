#! /usr/bin/env python

import numpy as np
import os
import time
import datetime
import data_helpers
import flags
import subprocess
from text_cnn import TextCNN
from collections import OrderedDict
import tensorflow as tf

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
    print("Max Document length: %s" % max_document_length)
    dir_path = os.path.dirname(os.path.realpath(__file__)) + "\\fasttext\\"
    print("FastText path: " + dir_path)
    p = subprocess.Popen([
        dir_path + "fasttext", "print-word-vectors", dir_path + "cc.en.300.bin"
    ],
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)
    x_list = []
    v_dict = [np.zeros((FLAGS.embedding_dim), dtype=np.float32)]
    w_dict = ["<%NONE%>"]
    w_freq = [0]
    for xt in x_text:
        cur_x = np.zeros((max_document_length), dtype=np.int)
        cur_w2v = GetFasttextArr(p, xt)
        winx = 0
        for w2v in cur_w2v:
            w = w2v[0] 
            v = w2v[1] 
            try:
                cinx = w_dict.index(w)
                cur_x[winx] = cinx
                w_freq[cinx] = w_freq[cinx] + 1
            except ValueError:
                w_dict.append(w)
                v_dict.append(v)
                w_freq.append(1)
                cur_x[winx] = len(w_dict) - 1
            winx = winx + 1
        x_list.append(cur_x)
    assert len(set(w_dict)) == len(w_dict), "Words Dictionary is not unique"
    np.savetxt(FLAGS.words_dic_file, np.transpose([w_dict, w_freq]), delimiter=',', fmt='%s')
    print("Done.")
    return np.stack(x_list), y, np.stack(v_dict)


def GetFasttextArr(proc_ft, in_str):
    proc_ft.stdin.write((in_str + ' <%EOL%>\n').encode('utf-8'))
    proc_ft.stdin.flush()
    is_eol = False
    w2v = []
    while not is_eol:
        l = proc_ft.stdout.readline().decode("utf-8")
        l = l.strip(' \t\n\r')
        is_eol = l.startswith('<%EOL%>')
        if not is_eol:
            lsplit = l.split(" ")
            snp = np.array(lsplit[1:])
            fnp = snp.astype(np.float32)
            w2v.append([lsplit[0], fnp])
    return w2v


def main(argv=None):
    x, y, embed_dict = preprocess()
    print(
        "Counts: x(rows, seg_max)=%s; y(rows, cols)=%s; embed_dict=%s"
        % (x.shape, y.shape, embed_dict.shape))
    np.savez(FLAGS.data_file, x=x, y=y, embed_dict=embed_dict)


if __name__ == '__main__':
    main()