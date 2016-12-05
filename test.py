import numpy as np

mat_1 = np.matrix([[4.5, -18.],[-17.,68.]])
mat_2 = np.matrix([[4.5, -17.],[-18.,68.]])
mat_3 = np.matrix([[1., -4.],[-4.,16.]])
I = np.matrix(np.identity(2))
r = 2./145

H = I-r*mat_1-r*mat_2+r*r*mat_2*mat_1+r*mat_3

print(H)
print(np.matrix(H))
