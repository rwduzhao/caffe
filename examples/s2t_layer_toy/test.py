#!/usr/bin/env python

import sys
sys.path.insert(0, '../python')
import caffe
import numpy as np

solver = caffe.SGDSolver('solver.prototxt')

data_blob = solver.net.blobs['data']
reshape_blob = solver.net.blobs['s2t']

data_blob.data[...] = np.arange(solver.net.blobs['data'].data.size).reshape(data_blob.data.shape)
solver.net.blobs['label'].data[...] = 1.

caffe.set_mode_cpu()
caffe.set_device(0)

niter = 1
for i in range(niter) :
  solver.step(1)
  print data_blob.data
  print reshape_blob.data.shape
  print reshape_blob.data[0][0]
  print reshape_blob.data[5][1]
  print reshape_blob.data[-1][-1]
