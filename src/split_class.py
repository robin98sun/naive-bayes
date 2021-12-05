#!/usr/bin/env python3

import argparse
import os
import random
import time

random.seed(int(time.time()))

def split_class(dir_path, strategy="alternative", training_percent=50, direction = "asc", shift = 0):
    if dir_path is None:
        return None
    if not os.path.isdir(dir_path):
        return None

    files = os.listdir(dir_path)
    if files is None or len(files) == 0:
        return None

    training_count_per_round = training_percent
    mode_for_alternating = 100
    if training_percent % 10 == 0:
        training_count_per_round = training_percent / 10
        mode_for_alternating = 10
    testing_count_per_round = mode_for_alternating - training_count_per_round

    result = {
        "train": [],
        "test": [],
    }
    for i in range(len(files)):
        filename = files[i]
        filepath = dir_path + "/" + filename
        if not os.path.isfile(filepath) or os.path.isdir(filepath):
            continue

        decision = "train"

        if strategy == "random":
            rnd = random.random()
            if rnd < 0.5:
                decision = "test"
        elif strategy == "sequential" and direction == "dsc":
            decision = "test"
        elif strategy == "alternative":
            if ((i+shift) % mode_for_alternating > training_count_per_round-1 and direction == "asc") \
                or ((i+shift) % mode_for_alternating <= testing_count_per_round-1 and direction == "dsc"):
                decision = "test"

        if decision == "train" and \
            len(result["train"]) >= training_percent * len(files) / 100:
            decision = "test"
        elif decision == "test" and \
            len(result["test"]) >= (100-training_percent) * len(files) / 100:
            decision = "train"

        result[decision].append(filepath)

    return result


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description='Project 2 of Course CSE 6363')
    arg_parser.add_argument('--dir', type=str, help='the directory for a class')
    arg_parser.add_argument('--training-percent', type=int, help='the percentage of sample data for training')
    arg_parser.add_argument('--strategy', type=str, default="alternative", choices=["sequential", "random", "alternative"],
        help='the strategy of selecting samples for training')
    arg_parser.add_argument('--direction', type=str, default="asc", choices=["asc", "dsc"],
        help='in which direction to select samples for training and testing')
    arg_parser.add_argument('--shift', type=int, default=0, help='shift from the alternative')
    args = arg_parser.parse_args()
    
    print("dir: {}".format(args.dir))
    print("strategy: {}".format(args.strategy))
    print("training percentage: {}".format(args.training_percent))

    splited = split_class(args.dir, args.strategy, args.training_percent, args.direction, args.shift)
    print("training set size: {}, testing set size: {}".format(len(splited["train"]), len(splited["test"])))
