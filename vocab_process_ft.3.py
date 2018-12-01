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
from threading import Thread
import sqlite3

# Parameters
# ==================================================
tf.flags.DEFINE_string("data_src_dbf", "./../data.arc/AclImdb_proc.db",
                       "Dict and Src[x, y] vectors sqlite database file")

FLAGS = tf.flags.FLAGS


def preprocess():
    print("\nParameters:")
    fdict = tf.app.flags.FLAGS.flag_values_dict()
    for attr in sorted(fdict):
        print("{}={}".format(attr.upper(), fdict[attr]))
    print("")
    # Data Preparation
    # ==================================================
    # Load data
    print("Loading data...")
    v_dict = []
    conn = sqlite3.connect(FLAGS.data_src_dbf)
    # Build vocabulary
    for row in conn.execute('SELECT VectStr FROM Dict ORDER BY Inx'):
        splt = np.array(row[0].split(" "))
        conv = splt.astype(np.float32)
        v_dict.append(conv)
    # Build x y
    x_raw = []
    max_document_length = 0
    y = []
    for row in conn.execute(
            'SELECT DictInxsStr, ProcInfo FROM Src WHERE NOT (OriginalId LIKE "test/%")'
    ):
        splt = np.array(row[0].split(" "))
        conv = splt.astype(np.int)
        max_document_length = max(len(conv), max_document_length)
        x_raw.append(conv)
        splt = np.array(row[1].split(" "))
        conv = splt.astype(np.int)
        y.append(conv)
    print("Max Document length: %s" % max_document_length)
    if FLAGS.max_document_length < max_document_length:
        raise ValueError("max_document_length: (FLAGS) %s < (actual) %s" %
                         (FLAGS.max_document_length, max_document_length))
    max_document_length = FLAGS.max_document_length

    x_res = []
    for x in x_raw:
        cur_x = np.zeros((max_document_length), dtype=np.int)
        for inx in range(0, len(x)):
            cur_x[inx] = x[inx]
        x_res.append(cur_x)

    print("Done.")
    return np.stack(x_res), np.stack(y), np.stack(v_dict)


def build_dict(conn):
    # Build vocabulary
    v_dict = []
    for row in conn.execute('SELECT VectStr FROM Dict ORDER BY Inx'):
        splt = np.array(row[0].split(" "))
        conv = splt.astype(np.float32)
        v_dict.append(conv)
    return np.stack(v_dict)


def build_xy(conn, where_sql):
    # Build x y
    x_raw = []
    max_document_length = 0
    y = []
    for row in conn.execute('SELECT DictInxsStr, ProcInfo FROM Src ' +
                            where_sql):
        splt = np.array(row[0].split(" "))
        conv = splt.astype(np.int)
        max_document_length = max(len(conv), max_document_length)
        x_raw.append(conv)
        splt = np.array(row[1].split(" "))
        conv = splt.astype(np.int)
        y.append(conv)
    print("Max Document length: %s" % max_document_length)
    if FLAGS.max_document_length < max_document_length:
        raise ValueError("max_document_length: (FLAGS) %s < (actual) %s" %
                         (FLAGS.max_document_length, max_document_length))
    max_document_length = FLAGS.max_document_length
    # fill zeros
    x_res = []
    for x in x_raw:
        cur_x = np.zeros((max_document_length), dtype=np.int)
        for inx in range(0, len(x)):
            cur_x[inx] = x[inx]
        x_res.append(cur_x)
    print("Done: build '%s'" % where_sql)
    return np.stack(x_res), np.stack(y)


def main(_):
    print("\nParameters:")
    fdict = tf.app.flags.FLAGS.flag_values_dict()
    for attr in sorted(fdict):
        print("{}={}".format(attr.upper(), fdict[attr]))
    print("")
    print("Loading data '%s'..." % FLAGS.data_src_dbf)
    conn = sqlite3.connect(FLAGS.data_src_dbf)
    embed_dict = build_dict(conn)
    print("Embed_dict=%s" % str(embed_dict.shape))
    # train data
    x, y  = build_xy(conn, 'WHERE NOT (OriginalId LIKE "test/%")')
    print("%s: x(rows, seg_max)=%s; y(rows, cols)=%s" %
          (FLAGS.data_file, str(x.shape), str(y.shape)))
    np.savez(FLAGS.data_file, x=x, y=y, embed_dict=embed_dict)
    # test data
    x, y  = build_xy(conn, 'WHERE OriginalId LIKE "test/%"')
    print("%s: x(rows, seg_max)=%s; y(rows, cols)=%s" %
          (FLAGS.data_file_test, str(x.shape), str(y.shape)))
    np.savez(FLAGS.data_file_test, x=x, y=y, embed_dict=embed_dict)
    #
    print("Done: Loading data")


if __name__ == '__main__':
    tf.app.run()