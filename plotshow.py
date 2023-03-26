import matplotlib.pyplot as plt
import numpy as np
import time

# create a plot
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)
plt.plot(x, y)
plt.title('Sine Wave')
plt.xlabel('x')
plt.ylabel('y')

plt.show()