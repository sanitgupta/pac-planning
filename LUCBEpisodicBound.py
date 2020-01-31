from constants import *
import math
import numpy as np
import sys
import time
from util import bestTwoActions, UpperP, LowerP, iteratedConvergence

def LUCBBound(mdp, start_state=0, epsilon=4, delta=0.1, fileprint=1):
	global MAX_ITERATION_LIMIT, c
	iteration = 0
	it=0
	H = int((math.log(mdp.Vmax) + math.log(6.0/epsilon))/(1-mdp.discountFactor))
	initial_iterations = 1*mdp.numStates*mdp.numActions
	rewards_s_a_sprime = np.zeros((mdp.numStates,mdp.numActions,mdp.numStates))
	R_s_a = np.zeros((mdp.numStates,mdp.numActions))
	sampled_frequency_s_a = np.zeros((mdp.numStates,mdp.numActions))
	N_s_a_sprime = np.zeros((mdp.numStates,mdp.numActions,mdp.numStates))
	P = np.zeros((mdp.numStates,mdp.numActions,mdp.numStates))
	P_tilda = np.zeros((mdp.numStates,mdp.numActions,mdp.numStates))
	P_lower_tilda = np.zeros((mdp.numStates,mdp.numActions,mdp.numStates))
	VlowerMBAE = np.zeros((mdp.numStates))
	Vlower = np.zeros((mdp.numStates))
	Vstar = (mdp.Vmax/2)*np.ones((mdp.numStates))
	Vupper = mdp.Vmax*np.ones((mdp.numStates))
	Qlower = np.zeros((mdp.numStates,mdp.numActions))
	VupperMBAE = mdp.Vmax*np.ones((mdp.numStates))
	QlowerMBAE = np.zeros((mdp.numStates,mdp.numActions))
	Qstar = (mdp.Vmax/2)*np.ones((mdp.numStates,mdp.numActions))
	QupperMBAE = mdp.Vmax*np.ones((mdp.numStates,mdp.numActions))
	Qupper = mdp.Vmax*np.ones((mdp.numStates,mdp.numActions))
	final_policy = (-1)*np.ones((mdp.numStates), dtype=np.int)
	states_to_sample = range(mdp.numStates)
	colliding_values = np.zeros((mdp.numStates))
	converge_iterations = 10000
	epsilon_convergence = 1e-4
	is_converged = 0
	print("Vmax", mdp.Vmax)

	### Initial sampling for all state action pairs
	while it < initial_iterations:
		for state in range(mdp.numStates):
			for act in range(mdp.numActions):
				it+=1
				s_prime, r = mdp.simulate(state, act)
				rewards_s_a_sprime[state][act][s_prime] += r
				R_s_a[state][act] = (r + R_s_a[state][act]*sampled_frequency_s_a[state][act])/(sampled_frequency_s_a[state][act]+1)
				sampled_frequency_s_a[state][act] += 1
				N_s_a_sprime[state][act][s_prime] += 1

	### Calculating V, Q estimates thus far MBAE
	for internal in range(converge_iterations):
		oldQlowerMBAE = np.copy(QlowerMBAE[start_state])
		for state in range(mdp.numStates):
			for act in range(mdp.numActions):
				# Calculations for QupperMBAE and QlowerMBAE
				firstterm = np.sum(rewards_s_a_sprime[state][act])/sampled_frequency_s_a[state][act]
				secondterm = mdp.discountFactor*np.sum(VupperMBAE*(N_s_a_sprime[state][act]/sampled_frequency_s_a[state][act]))
				#secondterm = mdp.discountFactor*sum(VupperMBAE[ss]*N_s_a_sprime[state][act][ss]/sampled_frequency_s_a[state][act] for ss in range(mdp.numStates))  
				lower_secondterm = mdp.discountFactor*np.sum(VlowerMBAE*(N_s_a_sprime[state][act]/sampled_frequency_s_a[state][act]))
				#lower_secondterm = mdp.discountFactor*sum(VlowerMBAE[ss]*N_s_a_sprime[state][act][ss]/sampled_frequency_s_a[state][act] for ss in range(mdp.numStates))  
				thirdterm = mdp.Vmax*math.sqrt((math.log(c*mdp.numStates*mdp.numActions)-math.log(delta))/sampled_frequency_s_a[state][act])
				#QupperMBAE[state][act] = (float)(sum(rewards_s_a_sprime[state][act][ss] for ss in range(mdp.numStates))/sampled_frequency_s_a[state][act]) + secondterm + thirdterm
				QupperMBAE[state][act] = firstterm + secondterm + thirdterm
				QlowerMBAE[state][act] = firstterm + lower_secondterm - thirdterm

				# Calculation for Vstar
				# t = (float)N_s_a_sprime[state][act][stateprime]/sampled_frequency_s_a[state][act]
				# val = t*(rewards_s_a[state][act][stateprime]+mdp.discountFactor*Vstar[stateprime])

				# if(state==start_state and abs(VupperMBAE[state]-QupperMBAEmax)<epsilon_convergence):
				# 	VupperMBAE[state] = QupperMBAEmax
				# 	print "Stopping with ", internal, "initial internal iterations"
				# 	is_converged = 1
				# 	break
			VupperMBAE[state] = np.amax(QupperMBAE[state])
			VlowerMBAE[state] = np.amax(QlowerMBAE[state])

		if(np.linalg.norm(oldQlowerMBAE-QlowerMBAE[start_state])<=epsilon_convergence):
			print("Stopping with ", internal, "initial internal iterations")
			break

	if internal==converge_iterations:
			print("Used all iterations")
	
	print("Initial estimate of QupperMBAE found! Now sampling")

	sys.stdout = open(mdp.filename+'-lucbbound.txt', 'w+')
	ff = open(mdp.filename+'-lucbbound-samples.txt', 'w+')

	h=0
	state1 = start_state

	while iteration<MAX_ITERATION_LIMIT:
		max_collision_state = [sorted(states_to_sample,key=lambda x: colliding_values[x], reverse=True)[0]]
		# print "Sampling state ", max_collision_state[0]
		# print colliding_values
		# print "Sampling ", state1, "for this round"
		if(h%H==0):
			state1 = start_state
			h = 0
		else:
			state1 = nextstate
		actionsList = bestTwoActions(mdp, state1, Qlower, Qupper, Qstar)
		a = np.random.choice(actionsList)
		iteration += 1
		sampled_frequency_s_a[state1][a] += 1
		for t in range(1):
			s_prime, r = mdp.simulate(state1, a)
			nextstate = s_prime
			rewards_s_a_sprime[state1][a][s_prime] += r
			R_s_a[state][act] = (r + R_s_a[state][act]*sampled_frequency_s_a[state][act])/(sampled_frequency_s_a[state][act]+1)
			N_s_a_sprime[state1][a][s_prime] += 1

		## Calculating Q and V values
		for i in range(mdp.numStates):
			for j in range(mdp.numActions):
				if(sampled_frequency_s_a[i][j]>0):
					P_tilda[i][j] = UpperP(i,j,delta,N_s_a_sprime[i][j],mdp.numStates,Vupper,False)
					P_lower_tilda[i][j] = LowerP(i,j,delta,N_s_a_sprime[i][j],mdp.numStates,Vupper,False)

		Qupper, Vupper = iteratedConvergence(Qupper,R_s_a,P_tilda,mdp.discountFactor, epsilon, converge_iterations, epsilon_convergence)
		Qlower, Vlower = iteratedConvergence(Qlower,R_s_a,P_lower_tilda,mdp.discountFactor, epsilon, converge_iterations, epsilon_convergence)
		
		# Calculations for QupperMBAE and QlowerMBAE
		#### This involved a two for-loop and iterating convergence
		for internal in range(converge_iterations):
			oldQlowerMBAE = np.copy(QlowerMBAE[start_state])
			for state in range(mdp.numStates):
				for act in range(mdp.numActions):
				# Calculations for QupperMBAE and QlowerMBAE
					# Calculations for QupperMBAE and QlowerMBAE
					firstterm = np.sum(rewards_s_a_sprime[state][act])/sampled_frequency_s_a[state][act]
					secondterm = mdp.discountFactor*np.sum(VupperMBAE*(N_s_a_sprime[state][act]/sampled_frequency_s_a[state][act]))
					#secondterm = mdp.discountFactor*sum(VupperMBAE[ss]*N_s_a_sprime[state][act][ss]/sampled_frequency_s_a[state][act] for ss in range(mdp.numStates))  
					lower_secondterm = mdp.discountFactor*np.sum(VlowerMBAE*(N_s_a_sprime[state][act]/sampled_frequency_s_a[state][act]))
					star_secondterm = mdp.discountFactor*np.sum(Vstar*(N_s_a_sprime[state][act]/sampled_frequency_s_a[state][act]))
					#lower_secondterm = mdp.discountFactor*sum(VlowerMBAE[ss]*N_s_a_sprime[state][act][ss]/sampled_frequency_s_a[state][act] for ss in range(mdp.numStates))  
					thirdterm = mdp.Vmax*math.sqrt((math.log(c*(iteration**2)*mdp.numStates*mdp.numActions)-math.log(delta))/sampled_frequency_s_a[state][act])
					#QupperMBAE[state][act] = (float)(sum(rewards_s_a_sprime[state][act][ss] for ss in range(mdp.numStates))/sampled_frequency_s_a[state][act]) + secondterm + thirdterm
					QupperMBAE[state][act] = firstterm + secondterm + thirdterm
					QlowerMBAE[state][act] = firstterm + lower_secondterm - thirdterm
					Qstar[state][act] = firstterm + star_secondterm
					# Calculation for Vstar
					# t = (float)N_s_a_sprime[state][act][stateprime]/sampled_frequency_s_a[state][act]
					# val = t*(rewards_s_a[state][act][stateprime]+mdp.discountFactor*Vstar[stateprime])
				VupperMBAE[state] = np.amax(QupperMBAE[state])
				VlowerMBAE[state] = np.amax(QlowerMBAE[state])
				Vstar[state] = np.amax(Qstar[state])
			if(np.linalg.norm(oldQlowerMBAE-QlowerMBAE[start_state])<=epsilon_convergence):
				# print "Stopping with ", internal, "iterations"
				break

		count = 0
		if(iteration%10000==0):
			print(iteration, (QupperMBAE[start_state][acList[1]]-QlowerMBAE[start_state][acList[0]])/epsilon )
			np.savetxt(ff, sampled_frequency_s_a, delimiter=',')
			ff.write('\n')
			# print QupperMBAE
			# print iteration
		
		#### Check epsilon condition for all the states
		# for st in range(mdp.numStates):
		# 	acList = bestTwoActions(mdp, st, Qstar, QupperMBAE, Qstar)
		# 	# print "Comparing ",QupperMBAE[st][acList[1]], QlowerMBAE[st][acList[0]]
			 
		# 	if(QupperMBAE[st][acList[1]]-QlowerMBAE[st][acList[0]]<=epsilon):
		# 		# print "Setting action ", acList[0], "for state ", st
		# 		final_policy[st]=acList[0]
		# 		count+=1

		##### Updating the list of coliliding states
		states_to_sample = []
		for st in range(mdp.numStates):
			acList = bestTwoActions(mdp, st, Qlower, Qupper, Qstar)
			# colliding_values[st] = QupperMBAE[st][acList[1]]-QlowerMBAE[st][acList[0]]-epsilon
			##### Changing stopping condition to epsilon*(1-gamma)/2
			colliding_values[st] = Qupper[st][acList[1]]-Qlower[st][acList[0]]-epsilon*(1-mdp.discountFactor)/2
			# print colliding_values[st]
			if(colliding_values[st]>0):
				### this state is still colliding, add to sample states
				states_to_sample.append(st)

		#### Check epsilon condition for only starting state
		if(not (start_state in states_to_sample)):
		# if(count==mdp.numStates):
			acList = bestTwoActions(mdp, start_state, Qlower, Qupper, Qstar)
			print("Setting final_policy of ", start_state, " to", acList[0])
			final_policy[start_state] = acList[0]
			print("Iterations taken : ", iteration)
			print("Returning the policy :", final_policy)
			for i in range(mdp.numStates):
				if(final_policy[i]==-1):
					final_policy[i] = bestTwoActions(mdp,i,Qlower,Qupper, Qstar)[0]
			return final_policy

		h+=1


	for i in range(mdp.numStates):
		if(final_policy[i]==-1):
			final_policy[i] = bestTwoActions(mdp,i,Qlower,Qupper, Qstar)[0]
	return final_policy