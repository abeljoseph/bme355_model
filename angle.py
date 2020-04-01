import numpy as np
import matplotlib.pyplot as plt

arr = []
with open('data_files/shank_angle_interpolated.csv') as f:
    for line in f:
        arr.append([float(x) for x in list(str(line).strip().split(','))])

arr = np.array(arr)

diff = []
for i in range(0, len(arr)-1):
    dx = arr[:,0][i+1] - arr[:,0][i]
    dy = arr[:,1][i+1] - arr[:,1][i]
    diff.append(dy/dx)    
    
plt.scatter(arr[:,0], arr[:,1])
plt.scatter(arr[:,0][1:], diff)
plt.show()
