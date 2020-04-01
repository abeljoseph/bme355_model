import csv
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
    diff.append([arr[:,0][i], dy/dx])    

diff.append([0.35, diff[-1][1]])  # Copy last value of list to end
diff = np.array(diff)

plt.scatter(arr[:,0], arr[:,1])
plt.scatter(diff[:,0], diff[:,1])
plt.show()

f = open('data_files/shank_velocity_interpolated.csv', 'w', newline='')
with f:
    writer = csv.writer(f)

    for val in diff:
        writer.writerow(val)