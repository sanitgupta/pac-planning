from constants import *
import math
import numpy as np
import sys
from util import bestTwoActions, UpperP, LowerP, iteratedConvergence
from evaluatePolicy import evaluatePolicy

verbose=0

def RoundRobin(mdp, start_state=0, epsilon=4, randomseed=None, delta=0.1):
	global MAX_ITERATION_LIMIT, c
	if(randomseed is not None):
		np.random.seed(randomseed)
	iteration = 0
	it=0
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
	VupperMBAE = mdp.Vmax*np.ones((mdp.numStates))
	Vupper = mdp.Vmax*np.random.random([mdp.numStates])
	QlowerMBAE = np.zeros((mdp.numStates,mdp.numActions))
	Qlower = np.zeros((mdp.numStates,mdp.numActions))
	Qstar = (mdp.Vmax/2)*np.ones((mdp.numStates,mdp.numActions))
	QupperMBAE = mdp.Vmax*np.ones((mdp.numStates,mdp.numActions))
	Qupper = mdp.Vmax*np.random.random([mdp.numStates,mdp.numActions])
	final_policy = (-1)*np.ones((mdp.numStates), dtype=np.int)
	states_to_sample = range(mdp.numStates)
	colliding_values = np.zeros((mdp.numStates))
	is_converged = 0

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
				for s2 in range(mdp.numStates):
					P[state][act][s_prime] = (float)(N_s_a_sprime[state][act][s_prime])/sampled_frequency_s_a[state][act]

	### Calculating V, Q estimates thus far
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

		VupperMBAE[state] = np.amax(QupperMBAE[state])
		VlowerMBAE[state] = np.amax(QlowerMBAE[state])

	Qupper = np.copy(QupperMBAE)
	Qlower = np.copy(QlowerMBAE)

	if(verbose==0):
		outp = open(mdp.filename+'-rr' + str(randomseed) +'.txt', 'wb')
	ff = open(mdp.filename+'-rr-samples.txt', 'w+')

	while iteration<MAX_ITERATION_LIMIT:
		# print "Sampling state ", max_collision_state[0]
		# print colliding_values
		for state1 in range(mdp.numStates):
			# print "Sampling ", state1, "for this round"
			for act1 in range(mdp.numActions):
				iteration += 1
				sampled_frequency_s_a[state1][act1] += 1
				
				# Simluate the MDP with this state,action and update counts
				#### TRying 10 continuous simulations 
				for t in range(1):
					s_prime, r = mdp.simulate(state1, act1)
					rewards_s_a_sprime[state1][act1][s_prime] += r
					R_s_a[state][act] = (r + R_s_a[state][act]*sampled_frequency_s_a[state][act])/(sampled_frequency_s_a[state][act]+1)
					N_s_a_sprime[state1][act1][s_prime] += 1
					for s2 in range(mdp.numStates):
						P[state1][act][s_prime] = (float)(N_s_a_sprime[state1][act][s_prime])/sampled_frequency_s_a[state1][act]
				
				## Calculating Q and V values
				for i in range(mdp.numStates):
					for j in range(mdp.numActions):
						if(sampled_frequency_s_a[i][j]>0):
							P_tilda[i][j] = UpperP(i,j,delta,N_s_a_sprime[i][j],mdp.numStates,Vupper,False)
							P_lower_tilda[i][j] = LowerP(i,j,delta,N_s_a_sprime[i][j],mdp.numStates,Vlower,False)

				Qupper, Vupper = iteratedConvergence(Qupper,R_s_a,P_tilda,mdp.discountFactor, epsilon, converge_iterations, epsilon_convergence)
				Qlower, Vlower = iteratedConvergence(Qlower,R_s_a,P_lower_tilda,mdp.discountFactor, epsilon, converge_iterations, epsilon_convergence)

				# Calculations for QupperMBAE and QlowerMBAE
				#### This involved a two for-loop and iterating convergence
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

		count = 0
		# print iteration, (QupperMBAE[start_state][acList[1]]-QlowerMBAE[start_state][acList[0]])/epsilon, sampled_frequency_s_a
		if(iteration%100==0):
			for i in range(mdp.numStates):
				if(final_policy[i]==-1):
					final_policy[i] = bestTwoActions(mdp,i,QlowerMBAE,QupperMBAE, Qstar)[0]
			acList = bestTwoActions(mdp, start_state, Qlower, Qupper, Qstar)
			if(verbose==0):
				outp.write(str(iteration))
				outp.write('\t')
				outp.write(str(QupperMBAE[start_state][acList[1]]-QlowerMBAE[start_state][acList[0]]))#-epsilon*(1-mdp.discountFactor)/2 
				# print(str(QupperMBAE[start_state][acList[1]]-QlowerMBAE[start_state][acList[0]]))
				# print(iteration, QupperMBAE[start_state])
				# outp.write(str(evaluatePolicy(mdp, final_policy, start_state)))
				print(str(evaluatePolicy(mdp, final_policy, start_state)))
				outp.write('\n')
			else:
				print(iteration, (QupperMBAE[start_state][acList[1]]-QlowerMBAE[start_state][acList[0]]))
				# print iteration, (Qupper[start_state][acList[1]]-Qlower[start_state][acList[0]])

			np.savetxt(ff, sampled_frequency_s_a, delimiter=',')
			ff.write('\n')
			# print iteration

		#### Check epsilon condition for only starting state
		acList = bestTwoActions(mdp, start_state, Qlower, Qupper, Qstar)
		if(QupperMBAE[start_state][acList[1]]-QlowerMBAE[start_state][acList[0]]<epsilon*(1-mdp.discountFactor)/2 and iteration>50):
			print(QupperMBAE[start_state][acList[1]]-QlowerMBAE[start_state][acList[0]],"<",epsilon*(1-mdp.discountFactor)/2)
		# if(count==mdp.numStates):
			acList = bestTwoActions(mdp, start_state, QlowerMBAE, QupperMBAE, Qstar)
			a = open('final'+mdp.filename+'-rr.txt', 'a+')
			a.write(str(iteration)+'\n')
			a.close()
			print("Setting final_policy of ", start_state, " to", acList[0])
			final_policy[start_state] = acList[0]
			print("Iterations taken : ", iteration)
			print("Returning the policy :", final_policy)
			for i in range(mdp.numStates):
				if(final_policy[i]==-1):
					final_policy[i] = bestTwoActions(mdp,i,QlowerMBAE,QupperMBAE, Qstar)[0]
			return final_policy


	for i in range(mdp.numStates):
		if(final_policy[i]==-1):
			final_policy[i] = bestTwoActions(mdp,i,QlowerMBAE,QupperMBAE, Qstar)[0]
	return final_policy