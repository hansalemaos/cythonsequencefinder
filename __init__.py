import os
import subprocess
import sys
from functools import reduce
import numpy as np


def _dummyimport():
    import Cython


try:
    from .cythonseqs import find_sequence_cython
except Exception as e:
    cstring = r"""# distutils: language=c
# distutils: extra_compile_args=/openmp
# distutils: extra_link_args=/openmp
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: language_level=3


from cython.parallel cimport prange
cimport cython
import numpy as np
cimport numpy as np
import cython

ctypedef fused real:
    cython.bint
    cython.char
    cython.schar
    cython.uchar
    cython.short
    cython.ushort
    cython.int
    cython.uint
    cython.long
    cython.ulong
    cython.longlong
    cython.ulonglong
    cython.float
    cython.double
    cython.longdouble
    cython.floatcomplex
    cython.doublecomplex
    cython.longdoublecomplex
    cython.size_t
    cython.Py_ssize_t
    cython.Py_hash_t
    cython.Py_UCS4

cdef void getseq(real[:] arr,real[:] seq, cython.int[:] result, int index, int seqsize, int lenresult):
    cdef Py_ssize_t rindex
    cdef int addindex=index+1
    if index == 0:
        for rindex in prange(lenresult,nogil=True):
            if seq[index] == arr[rindex]:
                result[rindex] = 1
        getseq(arr, seq, result, index + 1, seqsize,lenresult)
    else:
        for rindex in prange(1,lenresult,nogil=True):
            if result[rindex - 1] !=index:
                continue
            if result[rindex]>0:
                continue
            if arr[rindex] == seq[index]:
                result[rindex] = addindex


    if index + 1 < seqsize:
        getseq(arr, seq, result, index + 1, seqsize,lenresult)

cpdef void find_sequence_cython(real[:] a, real[:] b, cython.int[:] r):
    cdef int sequence_size=b.shape[0]
    cdef int lenresult=a.shape[0]
    cdef int index = 0
    getseq(a, b, r, index, sequence_size,lenresult)

"""
    pyxfile = f"seqs.pyx"
    pyxfilesetup = f"seqscompiled_setup.py"

    dirname = os.path.abspath(os.path.dirname(__file__))
    pyxfile_complete_path = os.path.join(dirname, pyxfile)
    pyxfile_setup_complete_path = os.path.join(dirname, pyxfilesetup)

    if os.path.exists(pyxfile_complete_path):
        os.remove(pyxfile_complete_path)
    if os.path.exists(pyxfile_setup_complete_path):
        os.remove(pyxfile_setup_complete_path)
    with open(pyxfile_complete_path, mode="w", encoding="utf-8") as f:
        f.write(cstring)
    numpyincludefolder = np.get_include()
    compilefile = (
        """
	from setuptools import Extension, setup
	from Cython.Build import cythonize
	ext_modules = Extension(**{'py_limited_api': False, 'name': 'cythonseqs', 'sources': ['seqs.pyx'], 'include_dirs': [\'"""
        + numpyincludefolder
        + """\'], 'define_macros': [], 'undef_macros': [], 'library_dirs': [], 'libraries': [], 'runtime_library_dirs': [], 'extra_objects': [], 'extra_compile_args': [], 'extra_link_args': [], 'export_symbols': [], 'swig_opts': [], 'depends': [], 'language': None, 'optional': None})

	setup(
		name='cythonseqs',
		ext_modules=cythonize(ext_modules),
	)
			"""
    )
    with open(pyxfile_setup_complete_path, mode="w", encoding="utf-8") as f:
        f.write(
            "\n".join(
                [x.lstrip().replace(os.sep, "/") for x in compilefile.splitlines()]
            )
        )
    subprocess.run(
        [sys.executable, pyxfile_setup_complete_path, "build_ext", "--inplace"],
        cwd=dirname,
        shell=True,
        env=os.environ.copy(),
    )
    try:
        from .cythonseqs import find_sequence_cython
    except Exception as fe:
        sys.stderr.write(f'{fe}')
        sys.stderr.flush()


def np_search_sequence(a, seq, distance=1):
    r"""NumPy implementation for finding sequences in arrays.

        Args:
            a (numpy.ndarray): The input array.
            seq (numpy.ndarray): The sequence to search for.
            distance (int): The maximum distance between elements in the sequence.

        Returns:
            numpy.ndarray: Indices of the found sequences in the input array.

        """
    return np.where(reduce(lambda a,b:a & b, ((np.concatenate([(a == s)[i * distance:], np.zeros(i * distance, dtype=np.uint8)],dtype=np.uint8)) for i,s in enumerate(seq))))[0]

def find_seq(arr,seq,distance=1):
    r"""Main function for finding sequences with an option for distance between elements.

        Args:
            arr (numpy.ndarray): The input array.
            seq (numpy.ndarray): The sequence to search for.
            distance (int): The maximum distance between elements in the sequence.

        Returns:
            numpy.ndarray: Indices of the found sequences in the input array.

        """
    try:
        sequence_size = len(seq)
        r = np.zeros(arr.shape, dtype=np.int32)
        if distance==1:
            find_sequence_cython(arr, seq, r)
            return (np.where(r == sequence_size))[0] - ((sequence_size) - 1)

        else:
            allre = []
            for d in range(distance):
                v = arr[d::distance]
                r = np.zeros(v.shape, dtype=np.int32)
                find_sequence_cython(v, seq, r)
                allre.append((((np.where(r == sequence_size)[0]) * distance) - ((sequence_size * distance) - distance - d)))
            return np.concatenate(allre)
    except Exception as fe:
        sys.stderr.write(f'{fe} - trying with NumPy')
        sys.stderr.flush()
        return np_search_sequence(arr, seq, distance=distance)