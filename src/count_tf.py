#!/usr/bin/env python3

import argparse
import json

def validate_char(ch):
    if ch is None:
        return False

    ascii_num = ord(ch)

    if ascii_num >= 48 and ascii_num <= 57:
        return True

    if ascii_num >= 65 and ascii_num <= 90:
        return True

    if ascii_num >= 97 and ascii_num <= 122:
        return True
       
    return False 

def validate_word(word):
    return True

def clean_trim(word, trim_side = "right"):
    if word is None or len(word) == 0:
        return None
    wl = len(word)
    if trim_side == "right":
        if not validate_char(word[wl-1]):
            word = word[0:-1]
        return clean_trim(word, trim_side = "left")
    elif trim_side == "left":
        if not validate_char(word[0]):
            word = word[1:]
        if len(word) == 0:
            return None
        else:
            return word


# setup stop words
def setup_stop_words(stop_words_file):
    STOP_WORDS = {}
    with open(stop_words_file, 'r', encoding = "ISO-8859-1") as f:
        lines = f.readlines()
        for line_idx in range(len(lines)):
            line = lines[line_idx].strip()
            for word in line.split():
                if word.lower() not in STOP_WORDS:
                    STOP_WORDS[word.lower()] = 1
    return STOP_WORDS

# count term frequency

def count_tf(sample_file, stop_words_file = None, output_file = None):
    TF_DICT, STOP_WORDS, lines_count = {}, {}, 0
    if stop_words_file is not None:
        STOP_WORDS = setup_stop_words(stop_words_file)

    with open(sample_file, 'r', encoding = "ISO-8859-1") as f:
        lines = f.readlines()
        lines_count = len(lines)
        for line_idx in range(len(lines)):
            line = lines[line_idx].strip()
            for raw_word in line.split():
                word = raw_word.lower().strip()
                # Clean the word
                cleaned_word = clean_trim(word)
                while cleaned_word != word:
                    word = cleaned_word
                    cleaned_word = clean_trim(word)
                word = cleaned_word
                if word is None:
                    continue
                elif not validate_word(word):
                    continue

                # ignore stop words
                is_stop_word = (word.lower() in STOP_WORDS)
                if not is_stop_word:
                    if word not in TF_DICT:
                        TF_DICT[word] = 1
                    else:
                        TF_DICT[word] += 1

    if output_file is not None:
        with open(args.output, 'w') as f:
            f.write(json.dumps(TF_DICT, sort_keys = True, indent = 4))

    return TF_DICT, lines_count

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description='Project 2 of Course CSE 6363')
    arg_parser.add_argument('--data', type=str, help='the data file path')
    arg_parser.add_argument('--output', type=str, help='the output file path')
    arg_parser.add_argument('--stop-words', type=str, help='stop words file path')

    args = arg_parser.parse_args()

    print("data file: {}".format(args.data))
    print("output file: {}".format(args.output))
    print("stop words: {}".format(args.stop_words))
    count_tf(args.data, args.stop_words, args.output)
