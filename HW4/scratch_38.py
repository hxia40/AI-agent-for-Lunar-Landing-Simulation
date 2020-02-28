import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import gym

#
# range_end = 10000
# episode = 8000
# a_list = []
# for episode in range(range_end):
#     a = (1 - math.log(episode+1, 10) / math.log(range_end, 10)) * 2
#     a_list.append(a)
#
# plt.grid()
#
# plt.plot(range(range_end), a_list)
#
# plt.savefig('test.png')

# def encode(self,taxi_row, taxi_col, pass_loc, dest_idx):
#     # (5) 5, 5, 4
#     i = taxi_row
#     i *= 5
#     i += taxi_col
#     i *= 5
#     i += pass_loc
#     i *= 4
#     i += dest_idx
#     return i
#
#
# def decode(self, i):
#     out = []
#     out.append(i % 4)
#     i = i // 4
#     out.append(i % 5)
#     i = i // 5
#     out.append(i % 5)
#     i = i // 5
#     out.append(i)
#     assert 0 <= i < 5
#     return reversed(out)


if __name__ == "__main__":
    env_HW4 = gym.make('Taxi-v2')
    for i in range(1000000):
        print(env_HW4.reset())
