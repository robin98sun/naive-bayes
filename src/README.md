# Prerequisites

1. Linux/Unix
2. Python3
    *  numpy
3. Data file

# Directory structure

1. Suppose the `20 newsgroup corpus` is located at `./20_newsgroups`
2. `./results` directory is readily created

# To perform one round of :
```bash
./test.py --dir ./20_newsgroups \
    --strategy alternative \
    --direction asc \
    --training-percent 50 \
    --shift 1 \
    --term-threshold 0 \
    --stop-words ./stop_words-long.txt \
    --smooth-factor 0.1
```

# To estimate the optimal smoothing factor:
```bash
./cross_validation.sh # it will take 3 hours
./analyze_results.py --dir ./results --smooth-factor-base 10
```

