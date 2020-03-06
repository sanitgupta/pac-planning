from constants import *
import math
import numpy as np
from pulp import *
import time
from util import LowerP, UpperP, CalculateDelDelV, iteratedConvergence, bestTwoActions

verbose=0
use_mbae = False
plot_vstar = True

def ddvouu(mdp, start_state=0, epsilon=4, randomseed=None, delta=0.1):

	if(randomseed is not None):
		np.random.seed(randomseed)

	initial_iterations = 1*mdp.numStates*mdp.numActions
	### Estimate the horizon based on Fiechter
	
	c = 1
	it=0
	samples = 0

	first_term = mdp.numStates/(epsilon**2*(1-mdp.discountFactor)**4)
	second_term = math.log(mdp.numStates*mdp.numActions/(epsilon*(1-mdp.discountFactor)*delta))/(epsilon**2*(1-mdp.discountFactor)**4)
	m = c*(first_term+second_term)
	delta = delta/(mdp.numStates*mdp.numActions*m)
	print("Chosen value of m is :", m)
	N_s_a = np.zeros((mdp.numStates,mdp.numActions), dtype=np.int)
	N_s_a_sprime = np.zeros((mdp.numStates,mdp.numActions,mdp.numStates), dtype=np.int)
	P_s_a_sprime = np.zeros((mdp.numStates,mdp.numActions,mdp.numStates))
	P_tilda = np.zeros((mdp.numStates,mdp.numActions,mdp.numStates))
	P_lower_tilda = np.zeros((mdp.numStates,mdp.numActions,mdp.numStates))
	R_s_a = np.zeros((mdp.numStates,mdp.numActions))
	Qupper = mdp.Vmax*np.ones((mdp.numStates,mdp.numActions))
	QupperMBAE = mdp.Vmax*np.ones((mdp.numStates,mdp.numActions))
	Qlower = np.zeros((mdp.numStates,mdp.numActions))
	Qstar = (mdp.Vmax/2)*np.ones((mdp.numStates,mdp.numActions))
	QlowerMBAE = np.zeros((mdp.numStates,mdp.numActions))
	Vupper = mdp.Vmax*np.ones((mdp.numStates))
	VupperMBAE = mdp.Vmax*np.ones((mdp.numStates))
	Vlower = np.zeros((mdp.numStates))
	VlowerMBAE = np.zeros((mdp.numStates))
	Vstar = (mdp.Vmax/2)*np.ones((mdp.numStates))
	best_policy = (-1)*np.ones((mdp.numStates), dtype=np.int)
	deltadeltaV = np.zeros((mdp.numStates,mdp.numActions))
	discovered_states = set([start_state, 1, 2, 3, 4])

	## Initial sampling for all state action pairs
	### Is this needed?
	while it < initial_iterations:
		for state in range(mdp.numStates):
			for act in range(mdp.numActions):
				it+=1

				ss, rr = mdp.simulate(state, act)
				print("Sampling ", state, act, rr, ss)
				R_s_a[state][act] = (rr + R_s_a[state][act]*N_s_a[state][act])/(N_s_a[state][act]+1)
				N_s_a[state][act] += 1
				N_s_a_sprime[state][act][ss] += 1
				# P_s_a_sprime = np.copy(N_s_a_sprime)
				for s2 in range(mdp.numStates):
					P_s_a_sprime[state][act][s2] = (float)(N_s_a_sprime[state][act][s2])/N_s_a[state][act]
	samples += initial_iterations

	print(P_s_a_sprime)
	print("Completed initial iterations")

	if(verbose==0):
		outp = open(mdp.filename+'-ddv' + str(randomseed) +'.txt', 'wb')
	# sys.stdout = open(mdp.filename+'-ddv.txt', 'w+')
	ff = open(mdp.filename+'-ddv-samples.txt', 'w+')
	
	# print Qupper, Vupper
	current_state = start_state
	### Repeat forever
	while samples<MAX_ITERATION_LIMIT:
		# print Qupper[start_state], Qlower[start_state]
		for i in range(mdp.numStates):
			for j in range(mdp.numActions):
				if(N_s_a[i][j]>0):
					P_tilda[i][j] = UpperP(i,j,delta,N_s_a_sprime[i][j],mdp.numStates,Vupper,False)
					P_lower_tilda[i][j] = LowerP(i,j,delta,N_s_a_sprime[i][j],mdp.numStates,Vlower,False)

		##Calculate Q values
		Qupper, Vupper = iteratedConvergence(
			Qupper,
			R_s_a,
			P_tilda,
			mdp.discountFactor,
			epsilon,
			converge_iterations,
			epsilon_convergence
			)
		Qlower, Vlower = iteratedConvergence(
			Qlower,
			R_s_a,
			P_lower_tilda,
			mdp.discountFactor,
			epsilon,
			converge_iterations,
			epsilon_convergence
			)	

		current_state = start_state

		### Terminating condition
		if(use_mbae):
			acList = bestTwoActions(mdp, start_state, QlowerMBAE, QupperMBAE, Qstar)
			coll = QupperMBAE[start_state][acList[1]]-QlowerMBAE[start_state][acList[0]]-epsilon*(1-mdp.discountFactor)/2
		else:
			acList = bestTwoActions(mdp, start_state, Qlower, Qupper, Qstar)
			coll = Qupper[start_state][acList[1]]-Qlower[start_state][acList[0]]-epsilon*(1-mdp.discountFactor)/2
		# if(Vupper[start_state]-Vlower[start_state]<=epsilon and samples>50):
		if(coll<0 and samples>50):
			a = open('final'+mdp.filename+'-ddv.txt', 'a+')
			a.write(str(samples)+'\n')
			a.close()
			print(Qupper[start_state],Vupper[start_state], Vlower[start_state])
			policy_lower = np.argmax(Qlower, axis=1)
			print("Iteration number ", samples)
			print("Returning policy because of epsilon-convergence")
			print(policy_lower)
			# print(np.argmax(QupperMBAE, axis=1))
			# print(np.argmax(Qupper, axis=1))
			# print(np.argmax(QlowerMBAE, axis=1))
			# print(np.argmax(Qstar, axis=1))
			# print(policy_lower)
			return policy_lower

		## Caclulate deldelV for all states
		if(use_mbae):
			for st in list(discovered_states):
				for ac in range(mdp.numActions):
					#### Compute del del V
					deltadeltaV[st][ac] = CalculateDelDelV(
						st,
						ac,
						mdp,
						N_s_a_sprime,
						QupperMBAE,
						QlowerMBAE,
						VupperMBAE,
						VlowerMBAE,
						start_state,
						P_s_a_sprime,
						P_tilda,
						P_lower_tilda,
						R_s_a,
						epsilon,
						delta,
						converge_iterations,
						epsilon_convergence
						)
		else:
			for st in list(discovered_states):
				for ac in range(mdp.numActions):
					#### Compute del del V
					deltadeltaV[st][ac] = CalculateDelDelV(
						st,
						ac,
						mdp,
						N_s_a_sprime,
						Qupper,
						Qlower,
						Vupper,
						Vlower,
						start_state,
						P_s_a_sprime,
						P_tilda,
						P_lower_tilda,
						R_s_a,
						epsilon,
						delta,
						converge_iterations,
						epsilon_convergence
						)

		#### Simulate greedily wrt deldelV
		# print np.unravel_index(deltadeltaV.argmax(), deltadeltaV.shape)
		current_state, current_action = np.unravel_index(deltadeltaV.argmax(), deltadeltaV.shape)
		#time.sleep(0.1)
		#print(deltadeltaV)
		ss, rr = mdp.simulate(current_state, current_action)
		samples += 1
		print(samples, "Sampling ", current_state, current_action, rr, ss)
		print(VupperMBAE[start_state] - VlowerMBAE[start_state])

		#### Add received state to the set of discovered states
		#discovered_states.add(ss)
		#print(discovered_states)
		### Update believed model
		R_s_a[current_state][current_action] = (rr + R_s_a[current_state][current_action]*N_s_a[current_state][current_action])/(N_s_a[current_state][current_action]+1)
		N_s_a[current_state][current_action] += 1	
		N_s_a_sprime[current_state][current_action][ss] += 1
		
		for s2 in range(mdp.numStates):
			# print current_state, current_action, s2, N_s_a_sprime[current_state][current_action][s2], N_s_a[current_state][current_action]
			P_s_a_sprime[current_state][current_action][s2] = (float)(N_s_a_sprime[current_state][current_action][s2])/N_s_a[current_state][current_action]
		
		if(samples%100==0):
			if(use_mbae):
				acList = bestTwoActions(mdp, start_state, QlowerMBAE, QupperMBAE, Qstar)
			else:
				acList = bestTwoActions(mdp, start_state, Qlower, Qupper, Qstar)
			# if(verbose==0):
			# 	outp.write(str(samples))
			# 	outp.write('\t')
			# 	if(plot_vstar):
			# 		outp.write(str(Vstar[start_state]))
			# 	else:
			# 		if(use_mbae):
			# 			outp.write(str(QupperMBAE[start_state][acList[1]]-QlowerMBAE[start_state][acList[0]]))
			# 		else:
			# 			outp.write(str(Qupper[start_state][acList[1]]-Qlower[start_state][acList[0]]))
			# 	outp.write('\n')
			# 	if(use_mbae):
			# 		print(samples, (QupperMBAE[start_state][acList[1]]-QlowerMBAE[start_state][acList[0]]))
			# 	else:
			# 		print(samples, (Qupper[start_state][acList[1]]-Qlower[start_state][acList[0]]))
			# else:
			# 	print(samples, (QupperMBAE[start_state][acList[1]]-QlowerMBAE[start_state][acList[0]]))
			# np.savetxt(ff, N_s_a, delimiter=',')
			# ff.write('\n')

		### Calculating MBAE bounds
		for internal in range(converge_iterations):
			oldQlower = np.copy(QlowerMBAE[start_state])
			for state in range(mdp.numStates):
				for act in range(mdp.numActions):
					# Calculations for Qupper and Qlower
					firstterm = R_s_a[state][act]
					secondterm = mdp.discountFactor*np.sum(VupperMBAE*(N_s_a_sprime[state][act]/N_s_a[state][act]))
					#secondterm = mdp.discountFactor*sum(Vupper[ss]*N_s_a_sprime[state][act][ss]/N_s_a[state][act] for ss in range(mdp.numStates))  
					lower_secondterm = mdp.discountFactor*np.sum(VlowerMBAE*(N_s_a_sprime[state][act]/N_s_a[state][act]))
					star_secondterm = mdp.discountFactor*np.sum(Vstar*(N_s_a_sprime[state][act]/N_s_a[state][act]))
					#lower_secondterm = mdp.discountFactor*sum(Vlower[ss]*N_s_a_sprime[state][act][ss]/N_s_a[state][act] for ss in range(mdp.numStates))  
					thirdterm = mdp.Vmax*math.sqrt((math.log(c*(samples**2)*mdp.numStates*mdp.numActions)-math.log(delta))/N_s_a[state][act])
					#Qupper[state][act] = (float)(sum(rewards_s_a_sprime[state][act][ss] for ss in range(mdp.numStates))/N_s_a[state][act]) + secondterm + thirdterm
					QupperMBAE[state][act] = firstterm + secondterm + thirdterm
					QlowerMBAE[state][act] = firstterm + lower_secondterm - thirdterm
					Qstar[state][act] = firstterm + star_secondterm
					# Calculation for Vstar
					# t = (float)N_s_a_sprime[state][act][stateprime]/N_s_a[state][act]
					# val = t*(rewards_s_a[state][act][stateprime]+mdp.discountFactor*Vstar[stateprime])
				VupperMBAE[state] = np.amax(QupperMBAE[state])
				VlowerMBAE[state] = np.amax(QlowerMBAE[state])
				Vstar[state] = np.amax(Qstar[state])
			if(np.linalg.norm(oldQlower-QlowerMBAE[start_state])<=epsilon_convergence):
				#print("Stopping with ", internal, " iterations and ", samples, " samples")
				break

		# if(samples==initial_iterations+2):
		# 	Qupper = np.copy(QupperMBAE)
		# 	Qlower = np.copy(QlowerMBAE)

	print(best_policy)
	return best_policy
