import numpy as np

import spam

spam.print("Hello, this is C")

rows = 2
arr = np.arange(10).reshape((2, -1))
print(arr)
cols = arr.shape[1]
b = arr.astype(np.float32).tobytes()
print(b)
spam.print_array(b, rows, cols)
