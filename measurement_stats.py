import pandas as pd

amazon = pd.read_csv('timing_logs/amazon_times_cpu.csv')
amazon_grouped = amazon.groupby('layers_in_enclave')['enclave_time']
print((amazon_grouped.std()/amazon_grouped.mean()).max())

flowers = pd.read_csv('timing_logs/flowers_times_cpu.csv')
flowers_grouped = flowers.groupby('layers_in_enclave')['enclave_time']
print((flowers_grouped.std()/flowers_grouped.mean()).max())

mit = pd.read_csv('timing_logs/mit_times_cpu.csv')
mit_grouped = mit.groupby('layers_in_enclave')['enclave_time']
print((mit_grouped.std()/mit_grouped.mean()).max())
