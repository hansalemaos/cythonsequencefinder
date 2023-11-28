# Module for finding sequences in arrays using Cython and NumPy.

## pip install cythonsequencefinder

### Tested against Windows / Python 3.11 / Anaconda


### Cython (and a C/C++ compiler) must be installed to use the optimized Cython implementation.




```python
from time import perf_counter
import numpy as np
from cythonsequencefinder import np_search_sequence
from cythonsequencefinder import find_seq

seq = np.array([1, 2, 4, 5], dtype=np.int32)
sequence_size = len(seq)
arr = np.random.randint(low=0, high=10, size=1000000, dtype=np.int32)
start = perf_counter()
rax = find_seq(arr, seq, distance=1)
print(perf_counter() - start)
start = perf_counter()
rax2 = np_search_sequence(arr, seq, distance=1)
print(perf_counter() - start)
# 0.004291800003557 - cython
# 0.007874400005675852 - numpy

```