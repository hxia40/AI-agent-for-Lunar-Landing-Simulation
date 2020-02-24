import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


range_end = 10000
episode = 8000
a_list = []
for episode in range(range_end):
    a = (1 - math.log(episode+1, 10) / math.log(range_end, 10)) * 2
    a_list.append(a)

plt.grid()

plt.plot(range(range_end), a_list)

plt.savefig('test.png')


