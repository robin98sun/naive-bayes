#!/usr/bin/env python3

import argparse
import os
from split_class import split_class

def split_corpus(dir_path, strategy="alternative", training_percent=50, direction = "asc", shift = 0):
    if dir_path is None:
        return None
    if not os.path.isdir(dir_path):
        return None

    classes = os.listdir(dir_path)
    if classes is None or len(classes) == 0:
        return None

    result = {
        "train": {},
        "test": {},
    }

    print("splitting corpus...")

    for i in range(len(classes)):
        classname = classes[i]
        classpath = dir_path + "/" + classname
        if not os.path.isdir(classpath):
            continue
        splited = split_class(classpath, strategy, training_percent, direction, shift)
        if splited is not None:
            print("class: {}, path: {}, training set size: {}, testing set size: {}".format(classname, classpath, len(splited["train"]), len(splited["test"])))
            result["train"][classname] = splited["train"]
            result["test"][classname] = splited["test"]

    print("corpus is splitted\n")

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
    print("direction: {}".format(args.direction))
    print("training percentage: {}".format(args.training_percent))
    print("shift: {}".format(args.shift))

    splited = split_corpus(args.dir, args.strategy, args.training_percent, args.direction, args.shift)

