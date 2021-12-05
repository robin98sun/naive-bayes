#!/usr/bin/env python3

from train import vectorize_corpus
from count_tf import count_tf
import argparse
import numpy as np
from time import time
import math

from train import vectorize_corpus
from test import test




if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description='Project 2 of Course CSE 6363, vectorize_corpus')
    arg_parser.add_argument('--dir', type=str, help='the directory for a class')
    arg_parser.add_argument('--stop-words', type=str, help='stop words file path')
    arg_parser.add_argument('--term-threshold', type=float, help='minimum possibility for terms')
    arg_parser.add_argument('--use-tf', action='store_true', help='if use the term frequency to predict')
    args = arg_parser.parse_args()

    TRAINING_PERCENT = 50

    min_sum_mse = -1
    min_idx = 0
    min_factor = 0
    sum_mse_list = []
    for i in range(1,11):
        smooth_factor = 0.1 * i
        sum_mse = 0
        mse_list = [smooth_factor]
        for shift in range(10):
            term_set, term_vector, class_set, class_vector, training_matrix, test_set, sample_count_dict, training_duration = vectorize_corpus(
                args.dir, args.stop_words, "alternative", TRAINING_PERCENT, 
                "asc", shift, args.term_threshold, smooth_factor,
            )

            correct_dict, global_error_table, duration = test(term_set, term_vector, class_set, class_vector, training_matrix, test_set, sample_count_dict, args.stop_words, args.use_tf)

            mse = 0
            for classname in correct_dict:
                mse += (1-correct_dict[classname])**2
            mse /= len(correct_dict)
            mse_list.append(mse)
            print("smoothing factor: {}, shift: {}, mse: {}\n".format(smooth_factor, shift, mse))
            sum_mse += mse
        mse_list.append(sum_mse)
        print("smoothing factor: {}, list: {}, sum: {}\n".format(mse_list[0], mse_list[1:-1], mse_list[-1]))
        sum_mse_list.append(mse_list)
        if min_sum_mse < 0 or sum_mse < min_sum_mse:
            min_sum_mse = sum_mse
            min_idx = i
            min_factor = smooth_factor
    print("best performance comes from the smoothing factor {}".format(min_factor))
    print(sum_mse_list)






