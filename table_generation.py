# coding: utf-8
import pandas as pd

vgg16 = pd.read_csv('timing_logs/mit_times_cpu.csv').groupby('layers_in_enclave').mean()
vgg19 = pd.read_csv('timing_logs/flowers_times_cpu.csv').groupby('layers_in_enclave').mean()
text = pd.read_csv('timing_logs/amazon_times_cpu.csv').groupby('layers_in_enclave').mean()

vgg16_len = len(vgg16.index)
vgg19_len = len(vgg19.index)
text_len = len(text.index)

max_index = max([vgg16_len, vgg19_len, text_len])

for i in range(max_index):
    if i < vgg16_len:
        vgg16_tf = f"{vgg16['tf_time'][i]:.03f}"
        
        if i > 0:
            vgg16_ee = f"{vgg16['enclave_time'][i]:.03f}"
            vgg16_ei = f"{vgg16['enclave_setup_time'][i]:.03f}"
        else:
            vgg16_ee = '--'
            vgg16_ei = '--'

        vgg16_total_time = vgg16['tf_time'][i] + vgg16['enclave_time'][i] + vgg16['enclave_setup_time'][i]
        vgg16_total = f"{vgg16_total_time:.03f}"
    else:
        vgg16_tf = '--'
        vgg16_ee = '--'
        vgg16_ei = '--'
        vgg16_total = '--'

    if i < vgg19_len:
        vgg19_tf = f"{vgg19['tf_time'][i]:.03f}"
        
        if i > 0:
            vgg19_ee = f"{vgg19['enclave_time'][i]:.03f}"
            vgg19_ei = f"{vgg19['enclave_setup_time'][i]:.03f}"
        else:
            vgg19_ee = '--'
            vgg19_ei = '--'

        vgg19_total_time = vgg19['tf_time'][i] + vgg19['enclave_time'][i] + vgg19['enclave_setup_time'][i]
        vgg19_total = f"{vgg19_total_time:.03f}"
    else:
        vgg19_tf = '--'
        vgg19_ee = '--'
        vgg19_ei = '--'
        vgg19_total = '--'

    if i < text_len:
        text_tf = f"{text['tf_time'][i]:.03f}"
        
        if i > 0:
            text_ee = f"{text['enclave_time'][i]:.03f}"
            text_ei = f"{text['enclave_setup_time'][i]:.03f}"
        else:
            text_ee = '--'
            text_ei = '--'

        text_total_time = text['tf_time'][i] + text['enclave_time'][i] + text['enclave_setup_time'][i]
        text_total = f"{text_total_time:.03f}"
    else:
        text_tf = '--'
        text_ee = '--'
        text_ei = '--'
        text_total = '--'

    line = """
    %d &
    \\scriptsize %s & \\scriptsize %s & \\scriptsize %s & \\textbf{%s} &
    \\scriptsize %s & \\scriptsize %s & \\scriptsize %s & \\textbf{%s} &
    \\scriptsize %s & \\scriptsize %s & \\scriptsize %s & \\textbf{%s} 
    \\\\ """ % (i,
            vgg16_tf, vgg16_ee, vgg16_ei, vgg16_total,
            vgg19_tf, vgg19_ee, vgg19_ei, vgg19_total,
            text_tf, text_ee, text_ei, text_total)

    print(line)
