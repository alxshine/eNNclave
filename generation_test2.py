import lib.frontend_python as frontend_python
import numpy as np

if __name__ == '__main__':
    inputs = np.arange(5, dtype=np.float32)
    result_bytes = frontend_python.native_forward(inputs.tobytes(), 5, 5)
    result = np.frombuffer(result_bytes, dtype=np.float32)
    print(result)
