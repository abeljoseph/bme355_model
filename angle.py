import numpy as np
import matplotlib.pyplot as plt

arr = []
with open('data_files/shank_angle.csv') as f:
    for line in f:
        arr.append([float(x) for x in list(str(line).strip().split(','))])

arr = np.array(arr)
plt.scatter(arr[:,0], arr[:,1])
plt.show()

fit = np.polyfit(arr[:,0], arr[:,1], 100)
print(fit)