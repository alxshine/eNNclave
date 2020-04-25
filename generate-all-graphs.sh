#!/bin/bash

python tikz_generation.py --model models/mit_tuned_frozen.h5 --time timing_logs/mit_times_cpu.csv timing_logs/mit_times_gpu.csv --ymin -3 --ymax 3 > mit.tex
# python tikz_generation.py --model models/imdb.h5 --time timing_logs/imdb_times_cpu.csv timing_logs/imdb_times_gpu.csv --ymin -3 --ymax 1 > imdb.tex
# python tikz_generation.py --model models/mnist.h5 --time timing_logs/mnist_times_cpu.csv timing_logs/mnist_times_gpu.csv --ymin -4 --ymax 1 > mnist.tex
# python tikz_generation.py --model models/rotten_tomatoes.h5 --time timing_logs/rotten_times_cpu.csv timing_logs/rotten_times_gpu.csv --ymin -3 --ymax 1 > rotten.tex
python tikz_generation.py --model models/amazon_dense.h5 --time timing_logs/amazon_times_cpu.csv --ymin -3 --ymax 1 > amazon.tex
python tikz_generation.py --model models/flowers.h5 --time timing_logs/flowers_times_cpu.csv --ymin -3 --ymax 3 > flowers.tex
