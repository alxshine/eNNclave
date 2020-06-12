import numpy as np

def dump_array(name, a):
    print("float %s[] = {" % name, end='')
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            print(f'{a[i,j]:.03}', end='')
            if i < a.shape[0]-1 or j < a.shape[1]-1:
                print(',', end='')
        print()
    print('};')

rand_a = np.random.rand(3,3) - .5
dump_array('rand_a', rand_a)
rand_b = np.random.rand(3,3) - .5
dump_array('rand_b', rand_b)
rand_res = rand_a+rand_b
dump_array('rand_exp', rand_res)
