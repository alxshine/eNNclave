import pandas as pd
import numpy as np
from itertools import chain

import tikz_utils

datasets = ['imdb', 'mnist', 'rotten', 'mit']
ds_names = {'imdb': 'IMDB', 'mnist': 'MNIST', 'rotten': 'R.T.', 'mit': 'MIT67'}
width = 0.2
inter_distance = 1
intra_distance = 0.1
start = 0.5

times = {}
for d in datasets:
    cpu_times = pd.read_csv(f'timing_logs/{d}_times_cpu.csv')
    gpu_times = pd.read_csv(f'timing_logs/{d}_times_gpu.csv')

    cpu_avg = cpu_times.where(cpu_times.layers_in_enclave == 0).mean().tf_time
    gpu_avg = gpu_times.where(gpu_times.layers_in_enclave == 0).mean().tf_time

    times[d] = (cpu_avg,gpu_avg)

flattened = list(chain(*times.values()))
y_min = np.floor(np.log10(min(flattened)))
y_max = np.ceil(np.log10(max(flattened)))

print(tikz_utils.generate_y_axis(y_min, y_max))


# draw rectangles
ret = ''
for i, (d, (cpu_time, gpu_time)) in enumerate(times.items()):
    cpu_x = start + i*inter_distance - intra_distance/2
    ret += '\\newcommand{\\comp%s}{%%\n' % d
    ret += '\\draw[fill=color1] (%f,0) rectangle (%f, %f);\n' % (cpu_x-width, cpu_x, tikz_utils.calc_log_coord(cpu_time, y_min, y_max)*tikz_utils.Y_MAX)

    gpu_x = start + i*inter_distance + intra_distance/2
    ret += '\\draw[fill=color3] (%f,0) rectangle (%f, %f);\n' % (gpu_x, gpu_x+width, tikz_utils.calc_log_coord(gpu_time, y_min, y_max)*tikz_utils.Y_MAX)

    # show dataset beneath
    ret += '\\draw (%f,-0.5) node {\\scriptsize %s};\n' % (start + i*inter_distance, ds_names[d])

    ret += '}\n'

print(ret)

ret = ''
ret = '\\newcommand{\\xaxis}{\\draw (0,0) -- (%d,0);}' % (i+1)*inter_distance
print(ret)
