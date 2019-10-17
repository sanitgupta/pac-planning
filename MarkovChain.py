from constants import *
import math
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt 
from util import getPolicies, UpperP, LowerP, indexOfPolicy, delW, getAverageRewards, ErrorV
from util import itConvergencePolicy, getRewards, getProb, allOneNeighbours
from util import CalculateDelDelV, prob_step, getSampleCount

verbose = 0
## policyMethod = 0 : brute force method, = 1 : nearest neighbour approach
policyMethod = 0
plot_vstar = False
fixedPolicy = [1, 1, 1, 1, 1, 1]

# Possible choices for algo are use_ddv, episodic, uniform, greedyMBAE, greedyMBIE, mybest

def markovchain(mdp, start_state=0, epsilon=4, randomseed=None, algo="episodic", delta=0.1, bounds="MBAE"):

	if(randomseed is not None):
		np.random.seed(randomseed)
	policies = np.array(getPolicies(mdp.numStates, mdp.numActions))
	numPolicies = len(policies)
	print "Total policies: ", numPolicies
	H = int(math.log(epsilon/(2*mdp.Vmax*(1 - mdp.discountFactor)))/math.log(mdp.discountFactor))
	print "Chosen value of H is : ", H

	
	## Initializations
	it = 0
	samples = 0
	initial_iterations = 1 * mdp.numStates * mdp.numActions
	R_s_a = np.zeros((mdp.numStates, mdp.numActions))
	R_s_a_sprime = np.zeros((mdp.numStates, mdp.numActions, mdp.numStates))
	N_s_a_sprime = np.zeros((mdp.numStates, mdp.numActions, mdp.numStates), dtype=np.int)
	N_s_a = np.zeros((mdp.numStates, mdp.numActions), dtype=np.int)
	P_s_a_sprime = np.zeros((mdp.numStates, mdp.numActions, mdp.numStates))
	Qupper = mdp.Vmax*np.ones((numPolicies, mdp.numStates))
	QupperMBAE = mdp.Vmax*np.ones((numPolicies, mdp.numStates))
	Qlower = np.zeros((numPolicies, mdp.numStates))
	Qstar = (mdp.Vmax/2)*np.ones((numPolicies, mdp.numStates))
	QstarMBAE = (mdp.Vmax/2)*np.ones((numPolicies, mdp.numStates))
	QlowerMBAE = np.zeros((numPolicies, mdp.numStates))
	
	V_true = np.zeros(mdp.numStates)
	V_estimate = np.zeros(mdp.numStates)
	V_error = np.zeros(MAX_ITERATION_LIMIT)

	P_tilda = np.zeros((numPolicies, mdp.numStates,mdp.numStates))
	P_lower_tilda = np.zeros((numPolicies, mdp.numStates,mdp.numStates))
	VlowerMBAE = np.zeros((numPolicies, mdp.numStates))
	VupperMBAE = mdp.Vmax*np.ones((numPolicies, mdp.numStates))
	Vstar = (mdp.Vmax/2)*np.ones((numPolicies, mdp.numStates))
	discovered_states = set([start_state])
	deltadeltaV = np.zeros((mdp.numStates))
	state_dist = np.zeros((mdp.numStates))
	state_dist[start_state] = 1
	print(state_dist)

	while it < initial_iterations:
		for state in range(mdp.numStates):
			for act in range(mdp.numActions):
				it = it + 1
				ss, rr = mdp.simulate(state, act)
				print "Sampling", state, act, rr, ss
				R_s_a[state][act] = (rr + R_s_a[state][act] * N_s_a[state][act])/(N_s_a[state][act] + 1)
				R_s_a_sprime[state][act][ss] = rr
				N_s_a[state][act] = N_s_a[state][act] + 1
				N_s_a_sprime[state][act][ss] = N_s_a_sprime[state][act][ss] + 1
				for s2 in range(mdp.numStates):
					P_s_a_sprime[state][act][s2] = (float)(N_s_a_sprime[state][act][s2])/N_s_a[state][act]
	samples += initial_iterations

	if(algo=="use_ddv"):
		ff = open(mdp.filename+'-markovddv' + str(randomseed) +'.txt', 'wb')
	elif(algo=="episodic"):
		ff = open(mdp.filename+'-markoveps' + str(randomseed) +'.txt', 'wb')
	elif(algo=="uniform"):
		ff = open(mdp.filename+'-markovuni' + str(randomseed) +'.txt', 'wb')
	elif(algo=="greedyMBAE"):
		ff = open(mdp.filename+'-markovMBAE' + str(randomseed) +'.txt', 'wb')
	elif(algo=="greedyMBIE"):
		ff = open(mdp.filename+'-markovMBIE' + str(randomseed) +'.txt', 'wb')
	elif(algo=="mybest"):
		ff = open(mdp.filename+'-markovbest' + str(randomseed) +'.txt', 'wb')
	elif(algo=="runcertainty"):
		ff = open(mdp.filename+'-markovruncertainty' + str(randomseed) +'.txt', 'wb')
	elif(algo=="unc_contri"):
		ff = open(mdp.filename+'-markovunc_contri' + str(randomseed) +'.txt', 'wb')

	
	while samples<MAX_ITERATION_LIMIT/2:
		
		p = 0
		current_policy = fixedPolicy

		for i in range(mdp.numStates):
			# print "For state ", i, " doing UpperP"
			if(N_s_a[i][current_policy[i]]>0):
				P_tilda[p][i] = UpperP(
					i,
					current_policy[i],
					delta,
					N_s_a_sprime[i][current_policy[i]],
					mdp.numStates,
					Qupper[p],
					False
					)
				P_lower_tilda[p][i] = LowerP(
					i,
					current_policy[i],
					delta,
					N_s_a_sprime[i][current_policy[i]],
					mdp.numStates,
					Qlower[p],
					False
					)

		Qupper[p] = itConvergencePolicy(
			Qupper[p],
			getRewards(R_s_a, current_policy),
			P_tilda[p],
			mdp.discountFactor,
			epsilon,
			converge_iterations,
			epsilon_convergence
			)

		Qlower[p] = itConvergencePolicy(
			Qlower[p],
			getRewards(R_s_a, current_policy),
			P_lower_tilda[p],
			mdp.discountFactor,
			epsilon,
			converge_iterations,
			epsilon_convergence
			)	

		Qstar[p] = itConvergencePolicy(
			Qstar[p],
			getRewards(R_s_a, current_policy),
			getProb(P_s_a_sprime, current_policy),
			mdp.discountFactor,
			epsilon,
			converge_iterations,
			epsilon_convergence
			)	

		for internal in range(converge_iterations):
			
			oldQlowerMBAE = np.copy(QlowerMBAE[p][start_state])
			for state in range(mdp.numStates):
				act = current_policy[state]
				firstterm = R_s_a[state][act]
				secondterm = mdp.discountFactor*np.sum(VupperMBAE[p]*(P_s_a_sprime[state][act]))
				lower_secondterm = mdp.discountFactor*np.sum(VlowerMBAE[p]*(P_s_a_sprime[state][act]))
				star_secondterm = mdp.discountFactor*np.sum(Vstar[p]*(P_s_a_sprime[state][act]))
				thirdterm = mdp.Vmax*math.sqrt((math.log(c*(samples**2)*mdp.numStates*1)-math.log(delta))/N_s_a[state][act])
				QupperMBAE[p][state] = firstterm + secondterm + thirdterm
				QlowerMBAE[p][state] = firstterm + lower_secondterm - thirdterm
				QstarMBAE[p][state] = firstterm + star_secondterm
				VupperMBAE[p][state] = QupperMBAE[p][state]
				VlowerMBAE[p][state] = QlowerMBAE[p][state]
				Vstar[p][state] = QstarMBAE[p][state]
			if(np.linalg.norm(oldQlowerMBAE-QlowerMBAE[p][start_state])<=epsilon_convergence):
				break
		policy1Index = 0
		
		h = 0
		policy1 = fixedPolicy
		state = start_state
		# print "samples", samples
		if (samples%10000)<100:
			if(verbose==0):
				ff.write(str(samples))
				ff.write('\t')
				if(plot_vstar):
					ff.write(str(Vstar[policy1Index][start_state]))
				else:
					ff.write(str(QupperMBAE[policy1Index][start_state]-QlowerMBAE[policy1Index][start_state]))#-epsilon*(1-mdp.discountFactor)/2 
				print samples, QupperMBAE[policy1Index][start_state]-QlowerMBAE[policy1Index][start_state]
				ff.write('\n')
			else:
				print samples
				print QupperMBAE[:,start_state], QlowerMBAE[:,start_state]

		polList = [policy1Index]

		# print(R_s_a.shape)
		# print(mdp.rewards)
		# print(mdp.transitionProbabilities)
		# print(getAverageRewards(mdp.numStates, mdp.numActions, mdp.rewards, mdp.transitionProbabilities))

		# print(getRewards(R_s_a, current_policy).shape)

		# print(getRewards(getAverageRewards(mdp.numStates, mdp.numActions, mdp.rewards, mdp.transitionProbabilities), current_policy).shape)

		# print(mdp.transitionProbabilities.shape)
		# print(P_s_a_sprime.shape)

		# print(getProb(mdp.transitionProbabilities, current_policy))
		# print(getProb(P_s_a_sprime, current_policy))


		# V_true = itConvergencePolicy(V_true,
		# 	getRewards(getAverageRewards(mdp.numStates, mdp.numActions, mdp.rewards, mdp.transitionProbabilities), current_policy),
		# 	getProb(mdp.transitionProbabilities, current_policy),
		# 	mdp.discountFactor,
		# 	epsilon,
		# 	converge_iterations,
		# 	epsilon_convergence
		# 	)

		# V_estimate	= itConvergencePolicy(V_estimate,
		# 	getRewards(R_s_a, current_policy),
		# 	getProb(P_s_a_sprime, current_policy),
		# 	mdp.discountFactor,
		# 	epsilon,
		# 	converge_iterations,
		# 	epsilon_convergence
		# 	)

		#print(V_estimate)
		V_true = itConvergencePolicy(V_true,
			getRewards(getAverageRewards(mdp.numStates, mdp.numActions, mdp.rewards, mdp.transitionProbabilities), current_policy),
			getProb(mdp.transitionProbabilities, current_policy),
			mdp.discountFactor,
			epsilon,
			converge_iterations,
			epsilon_convergence
			)
	


		if(algo=="use_ddv"):
			## Caclulate V for all states
			for pnum in polList:
				policiesfddv = fixedPolicy
				# print "Getting DDV values"
				for st in list(discovered_states):
					ac = policiesfddv[st]
					#### Compute del del V
					deltadeltaV[st] = CalculateDelDelV(
						st,
						ac,
						mdp,
						N_s_a_sprime,
						QupperMBAE[pnum],
						QlowerMBAE[pnum],
						None,
						None,
						start_state,
						P_s_a_sprime,
						P_tilda[pnum],
						P_lower_tilda[pnum],
						R_s_a,
						epsilon,
						delta,
						converge_iterations,
						epsilon_convergence,
						policiesfddv
						)

				# print deltadeltaV
				cs = np.argmax(deltadeltaV)
				ca = policiesfddv[cs]
				# print deltadeltaV, cs, ca
				# print deltadeltaV, policy1, policy2
				# print "Found max state for DDV: ",cs,ca
				# time.sleep(0.1)
				ss, rr = mdp.simulate(cs, ca)
				# print "Policy is ", policiesfddv
				# print "Sampling ", cs, ca

				time.sleep(0.1)	
				samples = samples +  1
				discovered_states.add(ss)
				R_s_a[cs][ca] = (rr + R_s_a[cs][ca]*N_s_a[cs][ca])/(N_s_a[cs][ca]+1)
				N_s_a[cs][ca] += 1
				N_s_a_sprime[cs][ca][ss] += 1
				# P_s_a_sprime = np.copy(N_s_a_sprime)
				for s2 in range(mdp.numStates):
					P_s_a_sprime[cs][ca][s2] = (float)(N_s_a_sprime[cs][ca][s2])/N_s_a[cs][ca]
		
		elif(algo == "episodic"):
			while h<H:
				act = policy1[state]
				# print "------>",current_state, current_action
				ss, rr = mdp.simulate(state, act)
				# print "Sampling ", state, act
				samples+=1
				R_s_a[state][act] = (rr + R_s_a[state][act]*N_s_a[state][act])/(N_s_a[state][act]+1)
				N_s_a[state][act] += 1
				N_s_a_sprime[state][act][ss] += 1
				# P_s_a_sprime = np.copy(N_s_a_sprime)
				for s2 in range(mdp.numStates):
					P_s_a_sprime[state][act][s2] = (float)(N_s_a_sprime[state][act][s2])/N_s_a[state][act]
				state = ss
				h+=1

				V_error[samples] = ErrorV(mdp, V_true, V_estimate, R_s_a, P_s_a_sprime, current_policy, start_state, epsilon, converge_iterations, epsilon_convergence)

		elif(algo == "uniform"):
			for st in range(mdp.numStates):
				ac = fixedPolicy[st]
				ss, rr = mdp.simulate(st, ac)
				# print "Sampling ", st, ac
				samples += 1
				R_s_a[st][ac] = (rr + R_s_a[st][ac]*N_s_a[st][ac])/(N_s_a[st][ac]+1)
				N_s_a[st][ac] += 1
				N_s_a_sprime[st][ac][ss] += 1
				for s2 in range(mdp.numStates):
					P_s_a_sprime[st][ac][s2] = (float)(N_s_a_sprime[st][ac][s2])/N_s_a[st][ac]

				V_error[samples] = ErrorV(mdp, V_true, V_estimate, R_s_a, P_s_a_sprime, current_policy, start_state, epsilon, converge_iterations, epsilon_convergence)


		elif(algo == "runcertainty"):

			deltaW = np.zeros(mdp.numStates)
			mu = np.zeros(mdp.numStates)
			D = np.zeros(mdp.numStates)

			mu[start_state] = 1

			for t in range(H):
				D = D + (mdp.discountFactor**t) * mu
				mu = prob_step(mu, P_s_a_sprime, fixedPolicy)

			for st in range(mdp.numStates):
				#transition uncertainty for given s, pi(s)
				deltaW[st] = delW(st, fixedPolicy[st], delta, N_s_a_sprime[st][fixedPolicy[st]], mdp.numStates, False)

			st = np.argmax(deltaW * D)
			# if samples % 100 == 0:
			# 	print deltaW, D, deltaW * D, np.argmax(deltaW * D)
			ac = fixedPolicy[st]
			ss, rr = mdp.simulate(st, ac)
			# print "Sampling ", st, ac, rr, ss
			samples += 1
			R_s_a[st][ac] = (rr + R_s_a[st][ac]*N_s_a[st][ac])/(N_s_a[st][ac]+1)
			N_s_a[st][ac] += 1
			N_s_a_sprime[st][ac][ss] += 1
			for s2 in range(mdp.numStates):
				P_s_a_sprime[st][ac][s2] = (float)(N_s_a_sprime[st][ac][s2])/N_s_a[st][ac]

			V_error[samples] = ErrorV(mdp, V_true, V_estimate, R_s_a, P_s_a_sprime, current_policy, start_state, epsilon, converge_iterations, epsilon_convergence)

		elif(algo == "unc_contri"):
			mu = np.zeros(mdp.numStates)
			D = np.zeros(mdp.numStates)

			z_quantile = 2.0

			mu[start_state] = 1


			transitionEstimate = getProb(P_s_a_sprime, fixedPolicy)
			rewardEstimate = getRewards(R_s_a, fixedPolicy)
			
			for t in range(H):
				D = D + (mdp.discountFactor**t) * mu

				mu = np.dot(mu, transitionEstimate)

			V_esti = np.dot(rewardEstimate, D)

			V_uncertainty = np.zeros(mdp.numStates)

			for s in range(mdp.numStates):

				transitionEstimateUpper = np.copy(transitionEstimate)
				transitionEstimateLower = np.copy(transitionEstimate)

				for sprime in range(mdp.numStates):
					#Wilson score interval confidence bounds for each transition

					muTerm = (transitionEstimate[s][sprime] + (z_quantile**2)/(2*N_s_a[s][fixedPolicy[s]])) / (1 + (z_quantile**2)/N_s_a[s][fixedPolicy[s]])
					devTerm = (z_quantile / (1 + (z_quantile**2)/N_s_a[s][fixedPolicy[s]])) * math.sqrt( (transitionEstimate[s][sprime]*(1-transitionEstimate[s][sprime])/N_s_a[s][fixedPolicy[s]]) + (z_quantile**2)/(4*N_s_a[s][fixedPolicy[s]]**2))
					
					# print(transitionEstimate[s][sprime], muTerm, devTerm)
					# if s == 1:
					# 	print "mu", muTerm
					# 	print "dev", devTerm
					# 	print (z_quantile / (1 + (z_quantile**2)/N_s_a[s][fixedPolicy[s]]))
					# 	print z_quantile, N_s_a[s][fixedPolicy[s]]
					# 	print (transitionEstimate[s][sprime]*(1-transitionEstimate[s][sprime])/N_s_a[s][fixedPolicy[s]])
					# 	print (z_quantile**2)/(4*N_s_a[s][fixedPolicy[s]]**2)
					transitionEstimateUpper[s][sprime] = muTerm + devTerm
					transitionEstimateLower[s][sprime] = muTerm - devTerm

					#print(transitionEstimateUpper[s][sprime], transitionEstimateLower[s][sprime] )
					#print("_____________")
				# if samples > 7775 and samples < 9249:
				# 	print s
				# 	print N_s_a[s][fixedPolicy[s]]
				# 	print transitionEstimate
				# 	print transitionEstimateUpper 
				# 	print transitionEstimateLower

				upperD = np.zeros(mdp.numStates)
				lowerD = np.zeros(mdp.numStates)

				uppermu = np.zeros(mdp.numStates)
				uppermu[start_state] = 1
				lowermu = np.zeros(mdp.numStates)
				lowermu[start_state] = 1

				for t in range(H):
					upperD = upperD + (mdp.discountFactor**t) * uppermu

					uppermu = np.dot(uppermu, transitionEstimateUpper)

					lowerD = lowerD + (mdp.discountFactor**t) * lowermu

					lowermu = np.dot(lowermu, transitionEstimateLower)
				# if samples > 7775 and samples < 9249:
				# 	print "___________"
				# 	print upperD
				# 	print lowerD
				# 	print rewardEstimate
				# 	print(np.dot(rewardEstimate, upperD) - np.dot(rewardEstimate, lowerD))

				# print(V_uncertainty)
				V_uncertainty[s] = abs(np.dot(rewardEstimate, upperD) - np.dot(rewardEstimate, lowerD))

				#print(s, rewardEstimate, upperD, lowerD)
			# print "V_unc", V_uncertainty
			st = np.argmax(V_uncertainty)	

			ac = fixedPolicy[st]
			ss, rr = mdp.simulate(st, ac)
			
			# if samples > 7775 and samples < 9249:
			# 	print "Sampling ", st, ac, rr, ss
			samples += 1
			R_s_a[st][ac] = (rr + R_s_a[st][ac]*N_s_a[st][ac])/(N_s_a[st][ac]+1)
			N_s_a[st][ac] += 1
			N_s_a_sprime[st][ac][ss] += 1
			for s2 in range(mdp.numStates):
				P_s_a_sprime[st][ac][s2] = (float)(N_s_a_sprime[st][ac][s2])/N_s_a[st][ac]

			V_error[samples] = ErrorV(mdp, V_true, V_estimate, R_s_a, P_s_a_sprime, current_policy, start_state, epsilon, converge_iterations, epsilon_convergence)

			# if V_error[samples]/V_true[start_state] > 0.35 and V_error[samples]> V_error[samples-1]:
			# 	print samples

		elif(algo == "greedyMBAE"):

			st = max(range(mdp.numStates), key=lambda x: VupperMBAE[0][x]-VlowerMBAE[0][x])
			ac = fixedPolicy[st]
			ss, rr = mdp.simulate(st, ac)
			# print "Sampling ", st, ac
			samples += 1
			R_s_a[st][ac] = (rr + R_s_a[st][ac]*N_s_a[st][ac])/(N_s_a[st][ac]+1)
			N_s_a[st][ac] += 1
			N_s_a_sprime[st][ac][ss] += 1
			for s2 in range(mdp.numStates):
				P_s_a_sprime[st][ac][s2] = (float)(N_s_a_sprime[st][ac][s2])/N_s_a[st][ac]

		elif(algo == "greedyMBIE"):

			st = max(range(mdp.numStates), key=lambda x: Qupper[0][x]-Qlower[0][x])
			ac = fixedPolicy[st]
			ss, rr = mdp.simulate(st, ac)
			# print "Sampling ", st, ac
			samples += 1
			R_s_a[st][ac] = (rr + R_s_a[st][ac]*N_s_a[st][ac])/(N_s_a[st][ac]+1)
			N_s_a[st][ac] += 1
			N_s_a_sprime[st][ac][ss] += 1
			for s2 in range(mdp.numStates):
				P_s_a_sprime[st][ac][s2] = (float)(N_s_a_sprime[st][ac][s2])/N_s_a[st][ac]

		elif(algo == "mybest"):
			if(samples%10000<50):
				state_dist = np.zeros((mdp.numStates))
				state_dist[start_state] = 1
			N = getSampleCount(state_dist, N_s_a_sprime, QupperMBAE[policy1Index], QlowerMBAE[policy1Index], QstarMBAE[policy1Index])
			# print N
			for i in range(N):
				# print state_dist, samples, P_s_a_sprime
				# import pdb; pdb.set_trace()
				st = np.random.choice(np.arange(mdp.numStates), p=state_dist)
				ac = fixedPolicy[st]
				ss, rr = mdp.simulate(st, ac)
				# print "Sampling ", st, ac
				samples += 1
				R_s_a[st][ac] = (rr + R_s_a[st][ac]*N_s_a[st][ac])/(N_s_a[st][ac]+1)
				N_s_a[st][ac] += 1
				N_s_a_sprime[st][ac][ss] += 1
				for s2 in range(mdp.numStates):
					P_s_a_sprime[st][ac][s2] = (float)(N_s_a_sprime[st][ac][s2])/N_s_a[st][ac]
			state_dist = prob_step(state_dist, P_s_a_sprime, fixedPolicy)

		if (samples%1000)<100:
			if(QupperMBAE[policy1Index][start_state]-QlowerMBAE[policy1Index][start_state]-epsilon*(1-mdp.discountFactor)/2<0):
				print Qupper[policy1Index][start_state],Qstar[policy1Index][start_state],epsilon*(1-mdp.discountFactor)/2
				print "Epsilon condition reached at ",samples, " samples"
				return fixedPolicy
			else:
				# print QupperMBAE[policy2Index][start_state],QstarMBAE[policy1Index][start_state],epsilon*(1-mdp.discountFactor)/2
				pass
			# print "ends here"
	# print(mdp.numStates, mdp.numActions)
	# plt.plot(1 + np.arange(MAX_ITERATION_LIMIT/2)[mdp.numStates * mdp.numActions + 500:], V_error[mdp.numStates * mdp.numActions + 500: MAX_ITERATION_LIMIT/2]/ V_true[start_state])
	# plt.title('Uniform Sampling')
	# plt.xlabel('samples')
	# plt.ylabel('Error fraction in value function')
	# plt.show()

	print(algo, " ", V_true)
	print(algo, " ", V_estimate)
	ff.close()
	return V_error/V_true[start_state]

