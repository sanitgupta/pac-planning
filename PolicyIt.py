from constants import *
import math
import numpy as np
import time
from util import *
from evaluatePolicy import evaluatePolicy 

verbose = 0
## policyMethod = 0 : brute force method, = 1 : nearest neighbour approach
policyMethod = 1
plot_vstar = True

def policyIt(mdp, start_state=0, epsilon=4, randomseed=None, delta=0.1, bounds="MBAE", use_ddv=False, mc = False, us = False):

	if(randomseed is not None):
		np.random.seed(randomseed)
	policies = np.array(getPolicies(mdp.numStates, mdp.numActions))
	numPolicies = len(policies)
	counts = np.zeros((numPolicies))
	print(numPolicies)
	#H = int((math.log(mdp.Vmax) + math.log(6.0/epsilon))/(1-mdp.discountFactor))
	H = int(math.log(epsilon/(2*mdp.Vmax*(1 - mdp.discountFactor)))/math.log(mdp.discountFactor))
	
	print("Chosen value of H is : ", H)

	
	## Initializations
	it = 0
	samples = 0
	initial_iterations = 1*mdp.numStates*mdp.numActions
	R_s_a = np.zeros((mdp.numStates,mdp.numActions))
	N_s_a_sprime = np.zeros((mdp.numStates,mdp.numActions,mdp.numStates), dtype=np.int)
	N_s_a = np.zeros((mdp.numStates,mdp.numActions), dtype=np.int)
	P_s_a_sprime = np.zeros((mdp.numStates,mdp.numActions,mdp.numStates))
	QU = mdp.Vmax*np.ones((mdp.numStates, mdp.numActions))
	QL = np.zeros((mdp.numStates, mdp.numActions))


	Qupper = mdp.Vmax*np.ones((numPolicies, mdp.numStates))
	QupperMBAE = mdp.Vmax*np.ones((numPolicies, mdp.numStates))
	Qlower = np.zeros((numPolicies, mdp.numStates))
	Qstar = (mdp.Vmax/2)*np.ones((numPolicies, mdp.numStates))
	QstarMBAE = (mdp.Vmax/2)*np.ones((numPolicies, mdp.numStates))
	QlowerMBAE = np.zeros((numPolicies, mdp.numStates))
	P_tilda = np.zeros((numPolicies, mdp.numStates,mdp.numStates))
	P_lower_tilda = np.zeros((numPolicies, mdp.numStates,mdp.numStates))
	Vupper = np.zeros(mdp.numStates)
	Vlower = np.zeros(mdp.numStates)

	opt_pol = [1,1,1,1,1,1]
	V_true = np.zeros(mdp.numStates)
	V_true = itConvergencePolicy(V_true,
			getRewards(getAverageRewards(mdp.numStates, mdp.numActions, mdp.rewards, mdp.transitionProbabilities), opt_pol),
			getProb(mdp.transitionProbabilities, opt_pol),
			mdp.discountFactor,
			epsilon,
			converge_iterations,
			epsilon_convergence
			)
	delvv = np.zeros(MAX_ITERATION_LIMIT)

	print(V_true)
	VlowerMBAE = np.zeros((numPolicies, mdp.numStates))
	VupperMBAE = mdp.Vmax*np.ones((numPolicies, mdp.numStates))
	Vstar = (mdp.Vmax/2)*np.ones((numPolicies, mdp.numStates))
	discovered_states = set([start_state])
	deltadeltaV = np.zeros((mdp.numStates))

	stop = lambda i, err : i > 1000 or err < 1e-9

	#sampling all state action pairs
	while it < initial_iterations:
		for state in range(mdp.numStates):
			for act in range(mdp.numActions):
				it = it + 1
				ss, rr = mdp.simulate(state, act)
				R_s_a[state][act] = (rr + R_s_a[state][act]*N_s_a[state][act])/(N_s_a[state][act]+1)
				N_s_a[state][act] = N_s_a[state][act] + 1
				N_s_a_sprime[state][act][ss] = N_s_a_sprime[state][act][ss] + 1
				# P_s_a_sprime = np.copy(N_s_a_sprime)
				for s2 in range(mdp.numStates):
					P_s_a_sprime[state][act][s2] = (float)(N_s_a_sprime[state][act][s2])/N_s_a[state][act]
	samples += initial_iterations

	if(use_ddv):
		ff = open(f"logs/{mdp.filename}-policyddv{randomseed}.txt", 'w')
	else:
		ff = open(f"logs/{mdp.filename}-policy{randomseed}.txt", 'w')
	
	while samples<MAX_ITERATION_LIMIT:
		
		# print counts
		if(policyMethod == 0):
			for p in range(numPolicies):
				# print "Policy Number : ", p
				current_policy = policies[p]

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

				# computing all three versions of Q given current knowlege of transition and reward matrices

	
				# if samples == 2000:
				# 	print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
				# 	Qupper[p] = QPIBoundsSolver(mdp, current_policy, P_s_a_sprime, QU, N_s_a, delta, True, stop)
				# 	#print(p)
				# 	Qlower[p] = QPIBoundsSolver(mdp, current_policy, P_s_a_sprime, QL, N_s_a, delta, False, stop)	
				# 	Qstar[p] = itConvergencePolicy(Qstar[p],
				# 	getRewards(R_s_a, current_policy),
				# 	getProb(P_s_a_sprime, current_policy),
				# 	mdp.discountFactor,
				# 	epsilon,
				# 	converge_iterations,
				# 	epsilon_convergence
				# 	)
				# 	print(Qupper[p], Qlower[p], current_policy)

				# import pdb; pdb.set_trace()
				# print "mbie bounds calculated!"
				for internal in range(converge_iterations):
					
					oldQlowerMBAE = np.copy(QlowerMBAE[p][start_state])
					for state in range(mdp.numStates):
						# for act in range(mdp.numActions):
						act = current_policy[state]
							# Calculations for QupperMBAE and QlowerMBAE
						firstterm = R_s_a[state][act]
						# print VupperMBAE[p]
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
				# print VupperMBAE[p]

		
			# import pdb; pdb.set_trace()
			policy1Index = np.argmax(QstarMBAE[:,start_state])
			policy2choices = QupperMBAE[:,start_state].argsort()[::-1]
			if(policy2choices[0]==policy1Index):
				policy2Index = policy2choices[1]
			else:
				policy2Index = policy2choices[0]
			# print "policyiniex", QstarMBAE[:,start_state]
		#action switching policy iteration
		elif(policyMethod==1):
			# print "Choosing 2nd method for finding policy"
			p = np.random.randint(0,numPolicies)
			current_policy = np.copy(policies[p])
			while True:
				for internal in range(converge_iterations):
					oldQlowerMBAE = np.copy(QlowerMBAE[p][start_state])
					for state in range(mdp.numStates):
						# for act in range(mdp.numActions):
						act = policies[p][state]
							# Calculations for QupperMBAE and QlowerMBAE
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

				hasChanged = False

				for st in range(mdp.numStates):
					for ac in range(mdp.numActions):
						if(current_policy[st]==ac):
							continue
						else:	
							tempfirstterm = R_s_a[st][ac]
							tempsecondterm = mdp.discountFactor*np.sum(VupperMBAE[p]*(P_s_a_sprime[st][ac]))
							lower_secondterm = mdp.discountFactor*np.sum(VlowerMBAE[p]*(P_s_a_sprime[st][ac]))
							star_secondterm = mdp.discountFactor*np.sum(Vstar[p]*(P_s_a_sprime[st][ac]))
							tempthirdterm = mdp.Vmax*math.sqrt((math.log(c*(samples**2)*mdp.numStates*1)-math.log(delta))/N_s_a[st][ac])
							tempQupperMBAE = tempfirstterm + tempsecondterm + tempthirdterm
							tempQlowerMBAE = tempfirstterm + lower_secondterm - tempthirdterm
							tempQstarMBAE = tempfirstterm + star_secondterm
							tempVupperMBAE = tempQupperMBAE
							tempVlowerMBAE = tempQlowerMBAE
							tempVstar = tempQstarMBAE

						if(tempVstar>Vstar[p][st]):
						# if(tempVupperMBAE>VupperMBAE[p][st]):
							current_policy[st] = ac
							hasChanged = True
							break

					# if(hasChanged):
					# 	break


				if hasChanged:
					p = indexOfPolicy(current_policy,mdp.numStates,mdp.numActions)
					# print("Changing to ",current_policy, p)
				else:
					policy1Index = p
					# print("Found first best policy!",policy1Index)
					break

			p = np.random.randint(0,numPolicies)
			current_policy = np.copy(policies[p])
			while True:
				for internal in range(converge_iterations):
					oldQlowerMBAE = np.copy(QlowerMBAE[p][start_state])
					for state in range(mdp.numStates):
						# for act in range(mdp.numActions):
						act = policies[p][state]
							# Calculations for QupperMBAE and QlowerMBAE
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

				hasChanged = False

				for st in range(mdp.numStates):
					for ac in range(mdp.numActions):
						if(current_policy[st]==ac):
							continue
						else:	
							tempfirstterm = R_s_a[st][ac]
							tempsecondterm = mdp.discountFactor*np.sum(VupperMBAE[p]*(P_s_a_sprime[st][ac]))
							lower_secondterm = mdp.discountFactor*np.sum(VlowerMBAE[p]*(P_s_a_sprime[st][ac]))
							star_secondterm = mdp.discountFactor*np.sum(Vstar[p]*(P_s_a_sprime[st][ac]))
							tempthirdterm = mdp.Vmax*math.sqrt((math.log(c*(samples**2)*mdp.numStates*1)-math.log(delta))/N_s_a[st][ac])
							tempQupperMBAE = tempfirstterm + tempsecondterm + tempthirdterm
							tempQlowerMBAE = tempfirstterm + lower_secondterm - tempthirdterm
							tempQstarMBAE = tempfirstterm + star_secondterm
							tempVupperMBAE = tempQupperMBAE
							tempVlowerMBAE = tempQlowerMBAE
							tempVstar = tempQstarMBAE

						if(tempVupperMBAE>VupperMBAE[p][st]):
						# if(tempVupperMBAE>VupperMBAE[p][st]):
							current_policy[st] = ac
							hasChanged = True
							break

					# if(hasChanged):
					# 	break


				if hasChanged:
					p = indexOfPolicy(current_policy,mdp.numStates,mdp.numActions)
					# print("Changing to ",current_policy, p, "Vupper")
				else:
					policy3Index = p
					if(policy3Index==policy1Index):
						### Hill Climbing for second policy
						# print "Finding 2nd best policy"
						# print VupperMBAE[:,start_state]
						oneNeighbours = allOneNeighbours(policies[policy3Index], mdp.numActions)

						maxVupper = -float("inf")
						bestPolicyIndex = -1
						hasChanged = False
						for p1 in oneNeighbours:
							# print p1
							# print indexOfPolicy(p1,mdp.numStates,mdp.numActions)
							p1Index = indexOfPolicy(p1,mdp.numStates,mdp.numActions)

							for internal in range(converge_iterations):
								oldQlowerMBAE = np.copy(QlowerMBAE[p1Index][start_state])
								for state in range(mdp.numStates):
									# for act in range(mdp.numActions):
									act = p1[state]
										# Calculations for QupperMBAE and QlowerMBAE
									firstterm = R_s_a[state][act]
									secondterm = mdp.discountFactor*np.sum(VupperMBAE[p1Index]*(P_s_a_sprime[state][act]))
									lower_secondterm = mdp.discountFactor*np.sum(VlowerMBAE[p1Index]*(P_s_a_sprime[state][act]))
									star_secondterm = mdp.discountFactor*np.sum(Vstar[p1Index]*(P_s_a_sprime[state][act]))
									thirdterm = mdp.Vmax*math.sqrt((math.log(c*(samples**2)*mdp.numStates*1)-math.log(delta))/N_s_a[state][act])
									QupperMBAE[p1Index][state] = firstterm + secondterm + thirdterm
									QlowerMBAE[p1Index][state] = firstterm + lower_secondterm - thirdterm
									QstarMBAE[p1Index][state] = firstterm + star_secondterm
									VupperMBAE[p1Index][state] = QupperMBAE[p1Index][state]
									VlowerMBAE[p1Index][state] = QlowerMBAE[p1Index][state]
									Vstar[p1Index][state] = QstarMBAE[p1Index][state]
								if(np.linalg.norm(oldQlowerMBAE-QlowerMBAE[p1Index][start_state])<=epsilon_convergence):
									break

								if(VupperMBAE[p1Index][start_state]>maxVupper):
									bestPolicyIndex = p1Index
									maxVupper = VupperMBAE[p1Index][start_state]
									# print Vstar[0]
									hasChanged = True

						if hasChanged:
							p = bestPolicyIndex
							policy2Index = bestPolicyIndex
							# print("Second best policy ", policy2Index)
					else:
						policy2Index = policy3Index
						# print(policies)
						# print(policies[1])
						# print(policies[63])
						# print("Found second best policy!",policy2Index)
					break



		#policy iteration methods end
		# if samples == 1200:
		# 	import pdb
		# 	pdb.set_trace()
		QU = QBoundsSolver(mdp, P_s_a_sprime, QU, N_s_a, delta, True, stop)
		QL = QBoundsSolver(mdp, P_s_a_sprime, QL, N_s_a, delta, False, stop)

		VU = np.max(QU, axis=1)

		VL = np.max(QL, axis=1)
		delvv[samples] = (VU[0] - VL[0])/V_true[0]
		print("Samples: ", samples, " delV", delvv[samples])

		# print(P_s_a_sprime)

		h=0
		policy1 = policies[policy1Index]
		policy2 = policies[policy2Index]
		# print QlowerMBAE
		# print policy2
		# print QstarMBAE[:,start_state]
		state = start_state
		if (samples%1000)<1000:
			if(verbose==0):
				# print QupperMBAE[:,start_state]
				# print Qstar[:,start_state]
				ff.write(str(samples))
				ff.write('\t')
				if(plot_vstar):
					# ff.write(str(Vstar[policy1Index][start_state]))
					ff.write(str(evaluatePolicy(mdp, policy1, start_state)))
					print(evaluatePolicy(mdp, policy1, start_state))
					print(samples, policy1, policy2)
				else:
					ff.write(str(QupperMBAE[policy2Index][start_state]-QlowerMBAE[policy1Index][start_state]))#-epsilon*(1-mdp.discountFactor)/2 

				print(samples, QupperMBAE[policy2Index][start_state]-QlowerMBAE[policy1Index][start_state])
				print(N_s_a)
				ff.write('\n')
			else:
				print(samples)
				print(QupperMBAE[:,start_state], QlowerMBAE[:,start_state])
			# np.savetxt(ff, (policies[policy1Index]), fmt="%d")
		counts[policy1Index] += 1
		counts[policy2Index] += 1

		polList = [policy1Index, policy2Index]


		if(use_ddv):
			## Caclulate V for all states
			for pnum in polList:
				policiesfddv = policies[pnum]
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
				# print("Policy is ", policiesfddv)
				# print("Sampling ", cs, ca)

				# time.sleep(0.1)	
				samples = samples +  1
				discovered_states.add(ss)
				R_s_a[cs][ca] = (rr + R_s_a[cs][ca]*N_s_a[cs][ca])/(N_s_a[cs][ca]+1)
				N_s_a[cs][ca] += 1
				N_s_a_sprime[cs][ca][ss] += 1
				# P_s_a_sprime = np.copy(N_s_a_sprime)
				for s2 in range(mdp.numStates):
					P_s_a_sprime[cs][ca][s2] = (float)(N_s_a_sprime[cs][ca][s2])/N_s_a[cs][ca]

		elif(mc):

			#for policy 1
			deltaW = np.zeros(mdp.numStates)
			mu_upper = np.zeros((H+1, mdp.numStates))
			D_upper = np.zeros(mdp.numStates)

			for st in range(mdp.numStates):
				#transition uncertainty for given s, pi(s)
				deltaW[st] = delW(st, policy1[st], delta, N_s_a_sprime[st][policy1[st]], mdp.numStates, False)

			mu_upper[0][start_state] = 1

			for t in range(H):
				D_upper = D_upper + (mdp.discountFactor**t) * mu_upper[t]

				for s in range(mdp.numStates):
					for s_ in range(mdp.numStates):
						mu_upper[t+1][s] += mu_upper[t][s_] * min(P_s_a_sprime[s_][policy1[s_]][s] + deltaW[s_], 1)

			st = np.argmax(deltaW * D_upper)

			ac = policy1[st]

			ss, rr = mdp.simulate(st, ac)
			samples += 1
			#print("Sample No. ", samples, " = ", st, ac, rr, ss)
			# print(mu_upper)
			# print(deltaW)
			

			#R_s_a[state][act] = (rr + R_s_a[state][act]*N_s_a[state][act])/(N_s_a[state][act]+1)
			N_s_a[st][ac] += 1
			N_s_a_sprime[st][ac][ss] += 1
			# P_s_a_sprime = np.copy(N_s_a_sprime)
			for s2 in range(mdp.numStates):
				P_s_a_sprime[st][ac][s2] = (float)(N_s_a_sprime[st][ac][s2])/N_s_a[st][ac]


			#for policy 2
			deltaW = np.zeros(mdp.numStates)
			mu_upper = np.zeros((H+1, mdp.numStates))
			D_upper = np.zeros(mdp.numStates)

			for st in range(mdp.numStates):
				#transition uncertainty for given s, pi(s)
				deltaW[st] = delW(st, policy2[st], delta, N_s_a_sprime[st][policy2[st]], mdp.numStates, False)

			mu_upper[0][start_state] = 1

			for t in range(H):
				D_upper = D_upper + (mdp.discountFactor**t) * mu_upper[t]

				for s in range(mdp.numStates):
					for s_ in range(mdp.numStates):
						mu_upper[t+1][s] += mu_upper[t][s_] * min(P_s_a_sprime[s_][policy2[s_]][s] + deltaW[s_], 1)

			st = np.argmax(deltaW * D_upper)
			ac = policy2[st]

			ss, rr = mdp.simulate(st, ac)
			samples += 1
			#print("Sample No. ", samples, " = ", st, ac, rr, ss)

			#R_s_a[state][act] = (rr + R_s_a[state][act]*N_s_a[state][act])/(N_s_a[state][act]+1)
			N_s_a[st][ac] += 1
			N_s_a_sprime[st][ac][ss] += 1
			# P_s_a_sprime = np.copy(N_s_a_sprime)
			for s2 in range(mdp.numStates):
				P_s_a_sprime[st][ac][s2] = (float)(N_s_a_sprime[st][ac][s2])/N_s_a[st][ac]

			if samples % 2000 == 0:
				VUpi2 = QPIBoundsSolver(mdp, policy2, P_s_a_sprime, QU, N_s_a, delta, True, stop)
			
				VLpi1 = QPIBoundsSolver(mdp, policy1, P_s_a_sprime, QL, N_s_a, delta, False, stop)

				print("Policy-eps ", samples, VUpi2[start_state], VLpi1[start_state], VUpi2[start_state] - VLpi1[start_state])

				np.save(f"pi_ucb_riverf_{samples}", delvv)

		elif us:
			for st in range(mdp.numStates):
				for ac in range(mdp.numActions):
					ss, rr = mdp.simulate(st, ac)
					samples += 1
					#print("Sample No. ", samples, " = ", st, ac, rr, ss)
					# print(mu_upper)
					# print(deltaW)
					

					#R_s_a[state][act] = (rr + R_s_a[state][act]*N_s_a[state][act])/(N_s_a[state][act]+1)
					N_s_a[st][ac] += 1
					N_s_a_sprime[st][ac][ss] += 1

			for st in range(mdp.numStates):
				for ac in range(mdp.numActions):		
					for s2 in range(mdp.numStates):
						P_s_a_sprime[st][ac][s2] = (float)(N_s_a_sprime[st][ac][s2])/N_s_a[st][ac]


		else:
			while h<H:
				act = policy1[state]
				# print "------>",current_state, current_action
				ss, rr = mdp.simulate(state, act)
				samples+=1
				R_s_a[state][act] = (rr + R_s_a[state][act]*N_s_a[state][act])/(N_s_a[state][act]+1)
				N_s_a[state][act] += 1
				N_s_a_sprime[state][act][ss] += 1
				# P_s_a_sprime = np.copy(N_s_a_sprime)
				for s2 in range(mdp.numStates):
					P_s_a_sprime[state][act][s2] = (float)(N_s_a_sprime[state][act][s2])/N_s_a[state][act]
				state = ss
				h+=1

			h=0

			if samples%10000 > 0:
				np.save(f"pi_pi_riverf_{samples}", delvv)

			state = start_state
			# print "episode : "
			while h<H:
				# print state, 
				act = policy2[state]
				ss, rr = mdp.simulate(state, act)
				samples+=1
				R_s_a[state][act] = (rr + R_s_a[state][act]*N_s_a[state][act])/(N_s_a[state][act]+1)
				N_s_a[state][act] += 1
				N_s_a_sprime[state][act][ss] += 1
				# P_s_a_sprime = np.copy(N_s_a_sprime)
				for s2 in range(mdp.numStates):
					P_s_a_sprime[state][act][s2] = (float)(N_s_a_sprime[state][act][s2])/N_s_a[state][act]
				state = ss
				h+=1

		if (samples%1000)<1000:
			if(QupperMBAE[policy2Index][start_state]-QlowerMBAE[policy1Index][start_state]-epsilon*(1-mdp.discountFactor)/2<0):
				print(Qupper[policy2Index][start_state],Qstar[policy1Index][start_state],epsilon*(1-mdp.discountFactor)/2)
				print("Epsilon condition reached at ", samples, " samples")
				print(policy1)
				return(policy1)
			else:
				#print("Stop",QupperMBAE[policy2Index][start_state]-QlowerMBAE[policy1Index][start_state]-epsilon*(1-mdp.discountFactor)/2)
				# print(QupperMBAE[policy2Index][start_state],QstarMBAE[policy1Index][start_state],epsilon*(1-mdp.discountFactor)/2)
				pass
			# print "ends here"

	ff.close()
	return policy1

