
import matplotlib.pyplot as plt
import numpy as np

# plt.figure(1)
# x = [0, 0.1, 0.2, 0.3, 0.4]
# y = [0.743, 0.733, 0.736, 0.724, 0.743]
# plt.plot(x, y)
# plt.ylabel("Vector Extrema")
# plt.xlabel("Relevance")
# plt.yticks(np.arange(0.65, 0.90, 0.05))
# plt.xticks(np.arange(0.0, 0.5, 0.1))
# axes = plt.gca()
# axes.set_xlim([0, 0.40])
# axes.set_ylim([0.65, 0.81])
# plt.show()

# plt.figure(2)
# x = [1, 2, 3, 4, 5]
# y = [0.750, 0.725, 0.709, 0.701, 0.722]
# plt.plot(x, y)
# plt.ylabel("Vector Extrema")
# plt.xlabel("Beam Width")
# plt.yticks(np.arange(0.65, 0.90, 0.05))
# plt.xticks(np.arange(1, 6, 1))
# axes = plt.gca()
# axes.set_xlim([1, 5.1])
# axes.set_ylim([0.65, 0.81])
# plt.show()

# plt.figure(3)
# x = [0.6, 0.8, 1.0, 1.2, 1.4]
# y = [0.696, 0.720, 0.705, 0.732, 0.749]
# plt.plot(x, y)
# plt.ylabel("Vector Extrema")
# plt.xlabel("Temperature")
# plt.yticks(np.arange(0.65, 0.90, 0.05))
# plt.xticks(np.arange(0.4, 1.6, 0.2))
# axes = plt.gca()
# axes.set_xlim([0.6, 1.41])
# axes.set_ylim([0.65, 0.81])
# plt.show()


# The Base line graph

plt.figure(4)
x = [2600, 5000, 8000, 10500]
y1 = [0.84, 0.792, 0.793, 0.799]
y2 = [1640, 2737, 1689, 201]
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(x, y1, 'b-')
ax2.plot(x, y2, 'g-')

ax1.set_ylim([0.65, 0.85])
ax2.set_ylim([])

ax1.set_xlabel('Training Iterations')
ax1.set_ylabel('Vec. Extrema', color='b')
ax2.set_ylabel('Perplexity', color='g')

plt.show()

