#!/usr/bin/env python3

from train import vectorize_corpus
from count_tf import count_tf
import argparse
import numpy as np
from time import time
import math
import json

def get_prior_vector_from_dict(sample_count_dict, class_set):
    # calc prior probability for each class
    sum_count = 0
    for classname in sample_count_dict:
        sum_count += sample_count_dict[classname]

    prior_vector = np.zeros((len(sample_count_dict),1), dtype=float)

    for classname in sample_count_dict:
        class_idx = class_set[classname]
        prior_vector[class_idx] = sample_count_dict[classname]/sum_count

    return prior_vector

def vectorize_sample(tf, term_set, term_vector_size, use_tf_dist = False):
    term_vector = np.zeros((1, term_vector_size), dtype=float)
    for term in tf:
        if term in term_set:
            if use_tf_dist:
                term_vector[0][term_set[term]] = tf[term]
            else:
                term_vector[0][term_set[term]] = 1

    if use_tf_dist:
        sum_scaler = np.sum(term_vector, axis=1, keepdims=True)
        squeezed_sum = np.squeeze(sum_scaler)
        term_vector = term_vector / sum_scaler

    return term_vector
    

def predict_sample(sample_vector, training_matrix, class_vector, prior_vector):
    # predict_vector = np.dot(training_matrix, np.transpose(sample_vector))
    # print("sample vector shape: {}, training matrix shape: {}".format(sample_vector.shape, training_matrix.shape))
    intermediate_matrix = training_matrix * sample_vector
    log_matrix = np.log(intermediate_matrix, where=(intermediate_matrix!=0)) 
    # print("log matrix shape: {}".format(log_matrix.shape))
    predict_vector = np.sum(log_matrix, axis=1) + np.log(prior_vector)
    # print("predict vector shape: {}".format(predict_vector.shape))
    class_idx = np.argmax(predict_vector)
    classname = class_vector[class_idx]
    return classname

def test(term_set, term_vector, class_set, class_vector, training_matrix, test_set, sample_count_dict, stop_words_file, use_tf):
    start_time = time()
    term_vector_size = len(term_vector)
    correct_dict = {}
    global_error_table = {}
    prior_vector = get_prior_vector_from_dict(sample_count_dict, class_set)
    for classname in test_set:
        correct_count = 0
        test_sample_count = len(test_set[classname])
        error_table = {}
        for sample_file in test_set[classname]:
            tf, lines_count = count_tf(sample_file, stop_words_file)
            sample_vector = vectorize_sample(tf, term_set, term_vector_size, use_tf)
            predict = predict_sample(sample_vector, training_matrix, class_vector, prior_vector)

            if predict == classname:
                correct_count += 1
            else:
                if predict not in error_table:
                    error_table[predict] = 1
                else:
                    error_table[predict] += 1

        correct_dict[classname] = correct_count/test_sample_count
        
        global_error_table[classname] = None
        for i in range(len(class_vector)):
            cn = class_vector[i]
            if cn not in error_table:
                continue
            if global_error_table[classname] is None \
                or error_table[cn]/test_sample_count > global_error_table[classname][1]:
                global_error_table[classname] = (cn, error_table[cn]/test_sample_count)
        print("class [{}] correct: {} in {} samples, correct rate: {}, largest error: {}".format(classname, correct_count, test_sample_count, correct_dict[classname], global_error_table[classname]))

    end_time = time()
    duration = end_time - start_time
    print("consumed {} seconds to test\n".format(round(duration)))
    
    return correct_dict, global_error_table, duration


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
    arg_parser.add_argument('--use-tf', action='store_true', help='if use the term frequency to predict')
    arg_parser.add_argument('--smooth-factor', type=float, default=0.1, help='the smooth factor for MAP')
    arg_parser.add_argument('--output', type=str, help='output the precision dictionary to a file')
    args = arg_parser.parse_args()

    print("dir: {}".format(args.dir))
    print("strategy: {}".format(args.strategy))
    print("direction: {}".format(args.direction))
    print("training percentage: {}".format(args.training_percent))
    print("shift: {}".format(args.shift))
    print("stop words: {}".format(args.stop_words))
    print("term threshold: {}".format(args.term_threshold))
    print("use term frequency to predict: {}".format(args.use_tf))
    print("smoothing factor: {}".format(args.smooth_factor))

    term_set, term_vector, class_set, class_vector, training_matrix, test_set, sample_count_dict, training_duration = vectorize_corpus(
        args.dir, args.stop_words, args.strategy, args.training_percent, 
        args.direction, args.shift, args.term_threshold, args.smooth_factor,
    )


    correct_dict, global_error_table, duration = test(term_set, term_vector, class_set, class_vector, training_matrix, test_set, sample_count_dict, args.stop_words, args.use_tf)

    if args.output is not None:
        with open(args.output, 'w') as f:
            f.write(json.dumps(correct_dict, sort_keys=True, indent=4))


    



