import numpy as np
import matplotlib.pyplot as plt

filename = 'shank_angle' + '_true.csv'
degree = 15

data = []
with open('data_files/'+filename) as f:
    for line in f:
        data.append([float(x) for x in list(str(line).strip().split(','))])

x = [i[0]-data[0][0] for i in data]
y = [i[1] for i in data]

xnew = np.linspace(0, data[-1][0]-data[0][0], 351)
eq = np.polyfit(x, y, degree)
ynew = 0
for i,j in zip(range(len(eq)-1, -1, -1), range(0, len(eq))):
    ynew += eq[j] * xnew**i

plt.figure()
plt.plot(x, y, label='Original Data')
plt.plot(xnew, ynew, '--', label='Polyfit')

plt.legend()
plt.show()

# Uncomment when happy with degree and ready to save interpolated data
np.savetxt(filename[:-9]+'_interpolated.csv', np.stack((xnew, ynew), axis=-1), fmt='%s', delimiter=',')