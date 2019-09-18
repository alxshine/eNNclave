import numpy as np

import interop.pymatutil as pymatutil

w1 = 3
h1 = 2
m1 = np.arange(6).reshape((h1, w1))
b1 = m1.astype(np.float32).tobytes()
print("m1:")
print(m1)

w2 = 2
h2 = 3
m2 = np.arange(6).reshape((h2, w2))
b2 = m2.astype(np.float32).tobytes()
print("m2:")
print(m2)

print("Result in python:")
pret = np.matmul(m1, m2)
print(pret)
btest = pret.astype(np.float32).tobytes()

print("Result in C:")
bret = pymatutil.multiply(b1, w1, h1, b2, w2, h2)
mret = np.frombuffer(bret, dtype=np.float32).reshape((2, 2))
print(mret)

print("Adding mret to mret")
bret2 = pymatutil.add(bret, 2, 2, bret, 2, 2)
mret2 = np.frombuffer(bret2, dtype=np.float32).reshape((2, 2))
print(mret2)

mrelu_in = np.copy(mret2)
mrelu_in[1, 1] = -1
print("Before relu:")
print(mrelu_in)
brelu = pymatutil.relu(mrelu_in.astype(np.float32).tobytes(), 2, 2)
mrelu = np.frombuffer(brelu, dtype=np.float32).reshape((2, 2))
print("After relu:")
print(mrelu)
