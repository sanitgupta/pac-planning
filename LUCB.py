from constants import *
import math
import numpy as np
import sys
import time
from util import bestTwoActions

verbose = 0

def LUCBStopping(mdp, start_state=0, epsilon=4, delta=0.1, fileprint=1):
	global MAX_ITERATION_LIMIT, c
	iteration = 0
	it=0
	initial_iterations = 1*mdp.numStates*mdp.numActions
	rewards_s_a_sprime = np.zeros((mdp.numStates,mdp.numActions,mdp.numStates))
	sampled_frequency_s_a = np.zeros((mdp.numStates,mdp.numActions))
	N_s_a_sprime = np.zeros((mdp.numStates,mdp.numActions,mdp.numStates))
	P = np.zeros((mdp.numStates,mdp.numActions,mdp.numStates))
	Vlower = np.zeros((mdp.numStates))
	Vstar = (mdp.Vmax/2)*np.ones((mdp.numStates))
	Vupper = mdp.Vmax*np.ones((mdp.numStates))
	Qlower = np.zeros((mdp.numStates,mdp.numActions))
	Qstar = (mdp.Vmax/2)*np.ones((mdp.numStates,mdp.numActions))
	Qupper = mdp.Vmax*np.ones((mdp.numStates,mdp.numActions))
	final_policy = (-1)*np.ones((mdp.numStates), dtype=np.int)
	states_to_sample = range(mdp.numStates)
	colliding_values = np.zeros((mdp.numStates))
	converge_iterations = 10000
	epsilon_convergence = 1e-4
	is_converged = 0
	print "Vmax", mdp.Vmax

	### Initial sampling for all state action pairs
	while it < initial_iterations:
		for state in range(mdp.numStates):
			for act in range(mdp.numActions):
				it+=1
				s_prime, r = mdp.simulate(state, act)
				rewards_s_a_sprime[state][act][s_prime] += r
				sampled_frequency_s_a[state][act] += 1
				N_s_a_sprime[state][act][s_prime] += 1

	### Calculating V, Q estimates thus far
	for internal in range(converge_iterations):
		oldQlower = np.copy(Qlower[start_state])
		for state in range(mdp.numStates):
			for act in range(mdp.numActions):
				# Calculations for Qupper and Qlower
				firstterm = np.sum(rewards_s_a_sprime[state][act])/sampled_frequency_s_a[state][act]
				secondterm = mdp.discountFactor*np.sum(Vupper*(N_s_a_sprime[state][act]/sampled_frequency_s_a[state][act]))
				#secondterm = mdp.discountFactor*sum(Vupper[ss]*N_s_a_sprime[state][act][ss]/sampled_frequency_s_a[state][act] for ss in range(mdp.numStates))  
				lower_secondterm = mdp.discountFactor*np.sum(Vlower*(N_s_a_sprime[state][act]/sampled_frequency_s_a[state][act]))
				star_secondterm = mdp.discountFactor*np.sum(Vstar*(N_s_a_sprime[state][act]/sampled_frequency_s_a[state][act]))
				#lower_secondterm = mdp.discountFactor*sum(Vlower[ss]*N_s_a_sprime[state][act][ss]/sampled_frequency_s_a[state][act] for ss in range(mdp.numStates))  
				thirdterm = mdp.Vmax*math.sqrt((math.log(c*mdp.numStates*mdp.numActions)-math.log(delta))/sampled_frequency_s_a[state][act])
				#Qupper[state][act] = (float)(sum(rewards_s_a_sprime[state][act][ss] for ss in range(mdp.numStates))/sampled_frequency_s_a[state][act]) + secondterm + thirdterm
				Qupper[state][act] = firstterm + secondterm + thirdterm
				Qlower[state][act] = firstterm + lower_secondterm - thirdterm
				Qstar[state][act] = firstterm + star_secondterm
				# Calculation for Vstar
				# t = (float)N_s_a_sprime[state][act][stateprime]/sampled_frequency_s_a[state][act]
				# val = t*(rewards_s_a[state][act][stateprime]+mdp.discountFactor*Vstar[stateprime])

				# if(state==start_state and abs(Vupper[state]-Quppermax)<epsilon_convergence):
				# 	Vupper[state] = Quppermax
				# 	print "Stopping with ", internal, "initial internal iterations"
				# 	is_converged = 1
				# 	break
			Vupper[state] = np.amax(Qupper[state])
			Vlower[state] = np.amax(Qlower[state])
			Vstar[state] = np.amax(Qstar[state])
		if(np.linalg.norm(oldQlower-Qlower[start_state])<=epsilon_convergence):
			print "Stopping with ", internal, "initial internal iterations"
			break

	if internal==converge_iterations:
			print "Used all iterations"
	
	print "Initial estimate of Qupper found! Now sampling"

	if(verbose==0):
		outp = open(mdp.filename+'-lucb.txt', 'wb')
	# sys.stdout = open(mdp.filename+'-lucb.txt', 'w+')
	ff = open(mdp.filename+'-lucb-samples.txt', 'w+')

	while iteration<MAX_ITERATION_LIMIT:
		max_collision_state = [sorted(states_to_sample,key=lambda x: colliding_values[x], reverse=True)[0]]
		# print "Sampling state ", max_collision_state[0]
		# print colliding_values
		for state1 in max_collision_state:
			# print "Sampling ", state1, "for this round"
			actionsList = bestTwoActions(mdp, state1, Qlower, Qupper, Qstar)
			for act1 in actionsList:
				iteration += 1
				sampled_frequency_s_a[state1][act1] += 1
				
				# Simluate the MDP with this state,action and update counts
				#### TRying 10 continuous simulations 
				for t in range(1):
					s_prime, r = mdp.simulate(state1, act1)
					rewards_s_a_sprime[state1][act1][s_prime] += r
					N_s_a_sprime[state1][act1][s_prime] += 1
				
				# Calculations for Qupper and Qlower
				#### This involved a two for-loop and iterating convergence
				for internal in range(converge_iterations):
					oldQlower = np.copy(Qlower[start_state])
					for state in range(mdp.numStates):
						for act in range(mdp.numActions):
						# Calculations for Qupper and Qlower
							# Calculations for Qupper and Qlower
							firstterm = np.sum(rewards_s_a_sprime[state][act])/sampled_frequency_s_a[state][act]
							secondterm = mdp.discountFactor*np.sum(Vupper*(N_s_a_sprime[state][act]/sampled_frequency_s_a[state][act]))
							#secondterm = mdp.discountFactor*sum(Vupper[ss]*N_s_a_sprime[state][act][ss]/sampled_frequency_s_a[state][act] for ss in range(mdp.numStates))  
							lower_secondterm = mdp.discountFactor*np.sum(Vlower*(N_s_a_sprime[state][act]/sampled_frequency_s_a[state][act]))
							#lower_secondterm = mdp.discountFactor*sum(Vlower[ss]*N_s_a_sprime[state][act][ss]/sampled_frequency_s_a[state][act] for ss in range(mdp.numStates))  
							thirdterm = mdp.Vmax*math.sqrt((math.log(c*(iteration**2)*mdp.numStates*mdp.numActions)-math.log(delta))/sampled_frequency_s_a[state][act])
							#Qupper[state][act] = (float)(sum(rewards_s_a_sprime[state][act][ss] for ss in range(mdp.numStates))/sampled_frequency_s_a[state][act]) + secondterm + thirdterm
							Qupper[state][act] = firstterm + secondterm + thirdterm
							Qlower[state][act] = firstterm + lower_secondterm - thirdterm

							# Calculation for Vstar
							# t = (float)N_s_a_sprime[state][act][stateprime]/sampled_frequency_s_a[state][act]
							# val = t*(rewards_s_a[state][act][stateprime]+mdp.discountFactor*Vstar[stateprime])
						Vupper[state] = np.amax(Qupper[state])
						Vlower[state] = np.amax(Qlower[state])
					if(np.linalg.norm(oldQlower-Qlower[start_state])<=epsilon_convergence):
						# print "Stopping with ", internal, "iterations"
						break

		count = 0
		if(iteration%100==0):
			if (verbose==0):
				outp.write(str(iteration))
				outp.write('\t')
				outp.write(str(Qupper[start_state][acList[1]]-Qlower[start_state][acList[0]]))#-epsilon*(1-mdp.discountFactor)/2 
				outp.write('\n')
			else:
				print Qupper[start_state], Qlower[start_state]
				# print d_h_policy_s[0][start_state]-2/(1-mdp.discountFactor)
				# print samples, (QupperMBAE[start_state][acList[1]]-QlowerMBAE[start_state][acList[0]])-epsilon*(1-mdp.discountFactor)/2
			np.savetxt(ff, sampled_frequency_s_a, delimiter=',')
			ff.write('\n')
			# print iteration, (Qupper[start_state][acList[1]]-Qlower[start_state][acList[0]])/epsilon 
			# print Qupper
			# print iteration
		
		#### Check epsilon condition for all the states
		# for st in range(mdp.numStates):
		# 	acList = bestTwoActions(mdp, st, Qstar, Qupper)
		# 	# print "Comparing ",Qupper[st][acList[1]], Qlower[st][acList[0]]
			 
		# 	if(Qupper[st][acList[1]]-Qlower[st][acList[0]]<=epsilon):
		# 		# print "Setting action ", acList[0], "for state ", st
		# 		final_policy[st]=acList[0]
		# 		count+=1

		##### Updating the list of coliliding states
		states_to_sample = []
		for st in range(mdp.numStates):
			acList = bestTwoActions(mdp, st, Qlower, Qupper, Qstar)
			# colliding_values[st] = Qupper[st][acList[1]]-Qlower[st][acList[0]]-epsilon
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
			print "Setting final_policy of ", start_state, " to", acList[0] 
			final_policy[start_state] = acList[0]
			print "Iterations taken : ", iteration
			print "Returning the policy :", final_policy
			for i in range(mdp.numStates):
				if(final_policy[i]==-1):
					final_policy[i] = bestTwoActions(mdp,i,Qlower,Qupper, Qstar)[0]
			return final_policy


	for i in range(mdp.numStates):
		if(final_policy[i]==-1):
			final_policy[i] = bestTwoActions(mdp,i,Qlower,Qupper, Qstar)[0]
	return final_policy