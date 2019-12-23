import matplotlib.pyplot as plt
import numpy as np

with open('result.txt') as f:
    losses = f.read().splitlines()

losses = [eval(v) for v in losses]
x = np.arange(len(losses))

f = np.polyfit(x, losses, 32)
f = np.poly1d(f)

print(f)
smooth_losses = f(x)

plt.plot(losses, label='losses')
plt.plot(smooth_losses, label='smooth')
plt.legend()
plt.show()
