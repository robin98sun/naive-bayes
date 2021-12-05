#!/usr/bin/env python3

import argparse
import json
import os
import numpy as np

def analyze_results(sub_dir, factor_base):
    files = os.listdir(sub_dir)
    count_factors = 0
    count_shifts = 0
    for i in range(len(files)):
        filename = files[i]
        if ".json" not in filename or "precision" not in filename:
            continue
        filepath = sub_dir + "/" + filename
        
        file_indices = filename[10:-5]
        indices_pair = file_indices.split("_")
        
        if int(indices_pair[0]) > count_factors:
            count_factors = int(indices_pair[0])
        if int(indices_pair[1]) > count_shifts:
            count_shifts = int(indices_pair[1])
    count_shifts += 1
    print("count factors: {}, count shifts: {}\n".format(count_factors, count_shifts))
    
    MSE_Matrix = np.zeros((count_factors,count_shifts))
    class_count = 0
    for factor_idx in range(count_factors):
        smooth_factor = float(factor_idx+1)/10
        for shift_idx in range(count_shifts):
            shift = shift_idx
            filename = sub_dir + '/precision_{}_{}.json'.format(factor_idx+1, shift_idx)
            with open(filename, 'r') as f:
                precisions = json.load(f)
                class_count = len(precisions)
                mse = 0
                for classname in precisions:
                    mse += (1-float(precisions[classname]))**2
                mse /= class_count
                MSE_Matrix[factor_idx][shift_idx] = mse
    # MSE_Matrix = np.append(MSE_Matrix, np.zeros((10,1)), 1)
    # print(MSE_Matrix.shape)
    avg_mse_vactor = np.sum(MSE_Matrix, axis = 1)/count_shifts
    # print(avg_mse_vactor.shape)
    # print(avg_mse_vactor)
    min_idx = np.argmin(avg_mse_vactor, axis = 0)
    optimal_smoothing_factor = (min_idx+1)/factor_base
    print("the optimal smoothing factor is: {}\n".format(optimal_smoothing_factor))
    for factor_idx in range(count_factors):
        smooth_factor = float(factor_idx+1)/10
        line = "{} & {} ".format(smooth_factor, round(avg_mse_vactor[factor_idx],3))
        for shift_idx in range(count_shifts):
            line += "& {} ".format(round(MSE_Matrix[factor_idx][shift_idx],3))
        line += "\\\\"
        # print(line)
        # print("\\hline")

    classname_list = None
    line_cache = {}
    avg_precision_cache = {}
    overall_average_precision = 0
    for shift_idx in range(count_shifts):
        filename = sub_dir + '/precision_{}_{}.json'.format(min_idx+1, shift_idx)
        with open(filename, 'r') as f:
            precisions = json.load(f)
            if classname_list is None:
                classname_list = list(precisions.keys())
            for classname in classname_list:
                if classname not in line_cache:
                    line_cache[classname] = " "
                if classname not in avg_precision_cache:
                    avg_precision_cache[classname] = 0
                line = line_cache[classname]
                line += "& "+str(round(precisions[classname],2)) + " "
                line_cache[classname] = line
                avg_precision_cache[classname] += precisions[classname]

    for classname in classname_list:
        avg_precision = avg_precision_cache[classname]/count_shifts
        line_cache[classname] = classname + " & " + str(round(avg_precision,3)) + " " + line_cache[classname] + "\\\\"
        overall_average_precision += avg_precision

    overall_average_precision /= len(classname_list)
    print("overall average precision for {} classes is: {}".format(len(classname_list), round(overall_average_precision,3)))

    # for classname in classname_list:
    #     line = line_cache[classname]
    #     print(line)
    #     print("\\hline")


    return optimal_smoothing_factor, avg_mse_vactor, MSE_Matrix, overall_average_precision, avg_precision_cache


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description='Project 2 of Course CSE 6363')
    arg_parser.add_argument('--dir', type=str, help='the directory for the results')
    arg_parser.add_argument('--smooth-factor-base', type=int, default=10, help='the directory for the results')


    args = arg_parser.parse_args()

    analyze_results(args.dir, args.smooth_factor_base)

