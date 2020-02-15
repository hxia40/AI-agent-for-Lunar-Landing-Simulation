import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

nested = [[35,36,37],[34,35,36,37,38],[22,23,23,24]]
fully_nested = [list(zip(*[(ix+1,y) for ix,y in enumerate(x)])) for x in nested]
names = ['sublist%d'%(i+1) for i in range(len(fully_nested))]

for l in fully_nested:
    plt.plot(*l)
plt.xlim(0,5)
plt.xlabel("Indices")
plt.ylim(0,40)
plt.xlabel("Values")
plt.legend(names, fontsize=7, loc = 'upper left')
plt.savefig('test.png')