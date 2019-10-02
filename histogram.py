import matplotlib.pyplot as plt 
import numpy as np
import sys

def float_convert(x):
	print x
	return float(x)

plt.title("NUmber of samples of (s,a)")
plt.xlabel("(s,a)")
plt.ylabel("Number of samples")

# Number of data plots
print sys.argv[1]

for i in range(int(sys.argv[1])):

	with open(sys.argv[i+2]) as textFile:
		data1 = [list(map(float_convert, line.split(','))) for line in textFile]

	data1 = [item for sublist in data1 for item in sublist]

	data1 = data1/np.sum(data1)

	# data1=np.array(data1)
	plt.plot(data1, label=sys.argv[i+2])

# plt.yaxis.tick_right()
# print data[:,0]
# a = plt.plot(data1[:,1], data1[:,0]/1e5, label=sys.argv[1])
# b = plt.plot(data2[:,1], data2[:,0]/1e5, label=sys.argv[2])
# b = plt.plot(data3[:,1], data3[:,0]/1e5, label=sys.argv[3])
plt.legend()
# plt.legend([sys.argv[2]])
# a.yaxis.tick_right()
plt.show()