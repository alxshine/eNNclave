import numpy as np

import interop.pymatutil as pymatutil

r1 = 2
c1 = 3
m1 = np.arange(6).reshape((r1, c1))
b1 = m1.astype(np.float32).tobytes()
print("m1:")
print(m1)

r2 = 3
c2 = 2
m2 = np.arange(6).reshape((r2, c2))
b2 = m2.astype(np.float32).tobytes()
print("m2:")
print(m2)

print("Result in python:")
pret = np.matmul(m1, m2)
print(pret)
btest = pret.astype(np.float32).tobytes()

print("Result in C:")
bret = pymatutil.multiply(b1, r1, c1, b2, r2, c2)
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
