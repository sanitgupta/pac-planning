import matplotlib.pyplot as plt 
import numpy as np
import sys

def float_convert(x):
	return float(x)

plt.title("MDP Planning")
plt.xlabel("Target delV as a function of V*")
plt.ylabel("Number of simulator calls (10^4)")

# Number of data plots
print sys.argv[1]

for i in range(int(sys.argv[1])):

	with open(sys.argv[i+2]) as textFile:
		data1 = [list(map(float_convert, line.split())) for line in textFile]

	data1=np.array(data1)
	# plt.plot(data1[:,1], data1[:,0]/1e5, label=sys.argv[i+2])
	ind = str(sys.argv[i+2]).find('-')+1
	rind = str(sys.argv[i+2]).find('spe')
	plt.plot(data1[:,1], data1[:,0]/1e4, label=str(sys.argv[i+2])[ind:rind])
	plt.xlim(0, 70)

# plt.yaxis.tick_right()
# print data[:,0]
# a = plt.plot(data1[:,1], data1[:,0]/1e5, label=sys.argv[1])
# b = plt.plot(data2[:,1], data2[:,0]/1e5, label=sys.argv[2])
# b = plt.plot(data3[:,1], data3[:,0]/1e5, label=sys.argv[3])
plt.legend()
# plt.legend([sys.argv[2]])
# a.yaxis.tick_right()
plt.title("mdp-SixArms")
plt.show()
