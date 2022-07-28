from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

t = np.linspace(-1, 1, 201)
x = (np.sin(2 * np.pi * 0.75 * t * (1 - t) + 2.1) +
     0.1 * np.sin(2 * np.pi * 1.25 * t + 1) +
     0.18 * np.cos(2 * np.pi * 3.85 * t))
xn = x + np.random.randn(len(t)) * 0.08
b, a = signal.butter(3, 0.05)
zi = signal.lfilter_zi(b, a)
dss = zi
data = []
data3, _ = signal.lfilter(b, a, xn, zi=dss)
print(dss)
for i in xn:
    z, dss = signal.lfilter(b, a, [i], zi=dss)

    data.append(z)
#     data2.append(z2)
# print(data2)
data2 = signal.filtfilt(b, a, xn)

plt.plot(t, xn, 'b', alpha=0.75)
plt.plot(t, data, 'r--')
plt.plot(t, data2, 'g')
plt.plot(t, data3, 'y^', alpha=0.3)
plt.grid(True)
plt.show()

