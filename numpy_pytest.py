import numpy

a = numpy.array([[1,2,3],[4,5,6],[7,8,9]])
print(a[[0,1],0].shape)
print(a[[0,1]][:,[0]].shape)