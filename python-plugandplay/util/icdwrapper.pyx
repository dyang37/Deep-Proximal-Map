# distutils: language = c++
# distutils: sources = icd.cpp

# Cython interface file for wrapping the object
#
#

from libcpp.vector cimport vector

# c++ interface to cython
cdef extern from "icd.h":
  cdef cppclass icd:
      icd(vector[vector[double]], vector[vector[double]], int, double, double) except +
      vector[vector[double]] update(vector[vector[double]], vector[vector[double]])

# creating a cython wrapper class
cdef class Pyicd:
    cdef icd *thisptr      # hold a C++ instance which we're wrapping
    def __cinit__(self, vector[vector[double]] y_py, vector[vector[double]] h_py, int _K, double _lambd, double _sigw):
        self.thisptr = new icd(y_py,h_py,_K,_lambd,_sigw)
    def __dealloc__(self):
        del self.thisptr
    def update(self, x, xtilde):
        return self.thisptr.update(x, xtilde)
