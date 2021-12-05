#!/usr/bin/env python3

from split_corpus import split_corpus
from count_tf import count_tf
import argparse
import numpy as np
from time import time

def count_df(tf_dict, df_current = None): 
    df = {}

    if df_current is not None:
        df = df_current

    for term in tf_dict:
        if term not in df:
            df[term] = 1
        else:
            df[term] += 1

    return df

def calc_possibility(df, sample_count, threshold=0, term_set = None, smooth_factor = 0.1):
    term_possibilities = {}
    for term in df:
        tp = (df[term] + smooth_factor) / (sample_count + smooth_factor * 2)
        if tp > threshold:
            term_possibilities[term] = tp
            if term_set is not None:
                if term not in term_set:
                    term_set[term] = 1

    return term_possibilities

def select_features():
    pass


def vectorize_corpus(dir_path, stop_words_file = None, strategy="alternative", training_percent=50, direction = "asc", shift = 0, term_threshold = 0, smooth_factor = 0.1):

    start_time = time()
    splited = split_corpus(dir_path, strategy, training_percent, direction, shift)

    print("learning the training set...")
    sample_set = "train"
    term_set = {}
    training_set = {}
    sample_count_dict = {}
    for classname in splited[sample_set]:
        raw_df = {}
        sample_count = len(splited[sample_set][classname])
        sample_count_dict[classname] = sample_count
        for sample_file in splited[sample_set][classname]:
            tf, lines_count = count_tf(sample_file, stop_words_file)
            count_df(tf, raw_df)
        print("class: {}, training sample amount: {}, terms: {}".format(classname, sample_count, len(raw_df)))
        tp = calc_possibility(raw_df, sample_count, term_threshold, term_set, smooth_factor)
        training_set[classname] = tp

    term_vector = list(term_set.keys())
    for i in range(len(term_vector)):
        term_set[term_vector[i]]=i

    class_vector = list(training_set.keys())
    class_set = {}    
    for i in range(len(class_vector)):
        class_set[class_vector[i]]=i

    # translate training set into matrix(class_N x term_M)
    training_matrix = np.zeros((len(class_vector), len(term_vector)), dtype=float)
    for i in range(training_matrix.shape[0]):
        classname = class_vector[i]
        for j in range(training_matrix.shape[1]):
            term = term_vector[j]
            tp = 0
            if term in training_set[classname]:
                tp = training_set[classname][term]
            else:
                tp = (0 + smooth_factor) / (sample_count_dict[classname] + smooth_factor * 2)
            training_matrix[i][j] = tp

    print("training matrix shape: {}".format(training_matrix.shape))

    end_time = time()
    duration = end_time - start_time

    print("consumed {} seconds to train\n".format(round(duration, 2)))
    return term_set, term_vector, class_set, class_vector, training_matrix, splited["test"], sample_count_dict, duration


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description='Project 2 of Course CSE 6363, vectorize_corpus')
    arg_parser.add_argument('--dir', type=str, help='the directory for a class')
    arg_parser.add_argument('--training-percent', type=int, help='the percentage of sample data for training')
    arg_parser.add_argument('--strategy', type=str, default="alternative", choices=["sequential", "random", "alternative"],
        help='the strategy of selecting samples for training')
    arg_parser.add_argument('--direction', type=str, default="asc", choices=["asc", "dsc"],
        help='in which direction to select samples for training and testing')
    arg_parser.add_argument('--shift', type=int, default=0, help='shift from the alternative')
    arg_parser.add_argument('--stop-words', type=str, help='stop words file path')
    arg_parser.add_argument('--term-threshold', type=float, help='minimum possibility for terms')
    arg_parser.add_argument('--smooth-factor', type=float, default=0.1, help='the smooth factor for MAP')
    args = arg_parser.parse_args()

    print("dir: {}".format(args.dir))
    print("strategy: {}".format(args.strategy))
    print("direction: {}".format(args.direction))
    print("training percentage: {}".format(args.training_percent))
    print("shift: {}".format(args.shift))
    print("stop words: {}\n".format(args.stop_words))

    vectorize_corpus(args.dir, args.stop_words, args.strategy, args.training_percent, args.direction, args.shift, args.term_threshold, args.smooth_factor)
