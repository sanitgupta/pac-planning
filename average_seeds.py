import sys
import numpy as np
from constants import seeds

print "MDP name is : ", sys.argv[1]
print "Method used is : ", sys.argv[2]

arr = []

min_length = None
for val in seeds:
	fname = sys.argv[1] + "-" + sys.argv[2] + str(val) + ".txt"
	tempa = np.loadtxt(fname)
	arr.append(tempa)
	if(min_length is None):
		min_length = tempa.shape[0]
	elif(min_length>tempa.shape[0]):
		min_length = tempa.shape[0]

arr = [x[:min_length] for x in arr]
arr = np.array(arr)

np.savetxt(sys.argv[1]+sys.argv[2]+"spe.txt", np.mean(arr, axis=0), newline="\n")


