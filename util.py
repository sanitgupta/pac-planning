from constants import *
import math
import pickle
import numpy as np
from pulp import *
import sys
from itertools import product

def pickle_save(obj, file_path):

    pickle.dump(obj, open(file_path, "wb"))

def CalculateValuePolicy(mdp, policy, H):
    total_reward = 0
    current_state = 0
    for i in range(H):
        current_action = policy[current_state]
        s, r = mdp.simulate(current_state, current_action)
        total_reward += r
        current_state = s

    return total_reward

def getSampleCount(prob_dist, N_s_a_sprime, Qupper, Qlower, Qstar):
    # print(np.dot(prob_dist,np.subtract(Qupper,Qlower)))
    return min(50,int(5*np.dot(prob_dist,np.subtract(Qupper,Qlower),Qstar)))

def prob_step(state_dist, P_s_a_sprime, fixedPolicy):
    return np.array([np.dot([P_s_a_sprime[x][fixedPolicy[x]][st] for x in range(len(state_dist))], state_dist) for st in range(len(state_dist))])

def bestTwoActions(mdp, state, Qlower, Qupper, Qstar):
    actionsList = []
    actionsList.append(np.argmax(Qstar[state]))
    a2 = np.argmax(Qupper[state])
    if(actionsList[0]!=a2):
        actionsList.append(a2)
    else:
        actionsList.append(Qupper[state].argsort()[::-1][1])

    # actionsList =  Qupper[state].argsort()[::-1][:2]
    return actionsList

def getBestPolicy(mdp, rewards, transitions):
    prob = LpProblem("MDP solver",LpMinimize)
    V_s = [0 for i in range(mdp.numStates)]
    for i in range(mdp.numStates):
        V_s[i] = LpVariable("Value for "+str(i))

    print(rewards)
    print(transitions)

    prob += lpSum(V_s[i] for i in range(mdp.numStates)), "Sum of V functions"

    for st in range(mdp.numStates):
        for ac in range(mdp.numActions):
            # rhs = lpSum(transitions[st][ac][sprime]*(rewards[st][ac][sprime]+gamma*V_s[sprime]) for sprime in range(numStates))
            prob += V_s[st] >= lpSum([transitions[st][ac][sprime]*(rewards[st][ac][sprime]+mdp.discountFactor*V_s[sprime]) for sprime in range(mdp.numStates)])
            # prob += V_s[st] >= sum(t*(r+2*v) for t,r,v,g in zip(transitions[st][ac],rewards[st][ac],V_s,gammaList))
    prob.writeLP("MDPmodel.lp")

    prob.solve()

    # print "Status:", LpStatus[prob.status]

    for v in prob.variables():
        print(v.name, "=", v.varValue)

    policy_final = [0 for i in range(mdp.numStates)]
    for st in range(mdp.numStates):
        maxV = -float("inf")
        maxAction = -1
        for ac in range(mdp.numActions):
            curVal = sum([transitions[st][ac][sprime]*(rewards[st][ac][sprime]+mdp.discountFactor*prob.variables()[sprime].varValue) for sprime in range(mdp.numStates)])
            if(curVal>maxV):
                maxAction = ac
                maxV = curVal

        policy_final[st] = maxAction    
    return policy_final

def hasConverged(a,b,epsilon=0.1):
    if(np.linalg.norm(np.subtract(a,b))<=epsilon):
        return True
    else:
        return False

def iteratedConvergence(Q, R, P, gamma, epsilon, maxIterations, eps_convergence):
    
    for i in range(maxIterations):
        # print Qupper[0]
        temp = np.copy(Q)
        Qmax_s = np.amax(Q,axis=1)
        # print "Norm ", np.linalg.norm(Qmax_s)
        Q = R + gamma*np.sum(P*Qmax_s, axis=2)
        V = np.amax(Q,axis = 1)
        if(hasConverged(Q, temp, eps_convergence)):
            break

    return Q, V

def itConvergencePolicy(Q, R, P, gamma, epsilon, maxIterations, eps_convergence):
    
    Qval = np.copy(Q)
    for i in range(maxIterations):
        # print Qupper[0]
        temp = np.copy(Qval)
        # Qmax_s = np.amax(Qupper,axis=1)
        # print "Norm ", np.linalg.norm(Qmax_s)
        # print P, Qval
        Qval = R + gamma*np.sum(P*Qval, axis=1)
        # Vupper = np.amax(Qupper,axis = 1)
        if(hasConverged(Qval, temp, eps_convergence)):
            break

    return Qval


#computes the d
def ErrorV(mdp, V_true, V_estimate, R_s_a, P_s_a_sprime, current_policy, start_state, epsilon, converge_iterations, epsilon_convergence):
        
        V_estimate  = itConvergencePolicy(V_estimate,
            getRewards(R_s_a, current_policy),
            getProb(P_s_a_sprime, current_policy),
            mdp.discountFactor,
            epsilon,
            converge_iterations,
            epsilon_convergence
            )
        #print(V_true, V_estimate)

        return (V_true[start_state] - V_estimate[start_state])

def wL1Confidence(N, delta, numStates):
    return math.sqrt(2*(math.log(2**numStates-2)-math.log(delta))/N)

def LowerP(state, action, delta, N_sprime, numStates, Vlower, good_turing, algo="mbie"):
    N_total = sum(N_sprime)

    P_hat_sprime = [(float)(N_sprime[i])/N_total for i in range(numStates)]
    P_tilda_sprime = [(float)(N_sprime[i])/N_total for i in range(numStates)]

    if(algo=="mbie"):
        delta_w = wL1Confidence(N_total,delta, numStates)/2
    else:
        delta_w = wL1Confidence(N_total,delta, numStates)/2

    if(good_turing):
        delta_w = min(wL1Confidence(N_total,delta/2, numStates)/2,(1+math.sqrt(2))*math.sqrt(math.log(2/delta)/N_total))

    ite=0

    while delta_w>0 and ite<100:
        recipient_states = [i for i in range(numStates) if P_tilda_sprime[i]<1]
        positive_states = [i for i in range(numStates) if P_tilda_sprime[i]>0]
        donor_s = max(positive_states, key=lambda x: Vlower[x])
        recipient_s = min(recipient_states, key=lambda x: Vlower[x])
        zeta = min(1-P_tilda_sprime[recipient_s], P_tilda_sprime[donor_s], delta_w)
        if(not P_tilda_sprime[donor_s]>sys.float_info.epsilon):
            break   
        P_tilda_sprime[donor_s] -= zeta
        P_tilda_sprime[recipient_s] += zeta

        delta_w -= zeta
        ite = ite + 1

    return P_tilda_sprime 

def delW(state, action, delta, N_sprime, numStates, good_turing):
    N_total = sum(N_sprime)
    delta_w = wL1Confidence(N_total,delta, numStates)/2
    return delta_w


def UpperP(state, action, delta, N_sprime, numStates, Vupper, good_turing, algo="mbie"):
    N_total = sum(N_sprime)

    P_hat_sprime = [(float)(N_sprime[i])/N_total for i in range(numStates)]
    P_tilda_sprime = [(float)(N_sprime[i])/N_total for i in range(numStates)]

    if(algo=="mbie"):
        delta_w = wL1Confidence(N_total, delta, numStates)/2
    else:
        #delta_w = wL1Confidence(N_total+1,delta, numStates)/2
        delta_w = wL1Confidence(N_total, delta, numStates)/2

    if(good_turing):
        delta_w = min(wL1Confidence(N_total,delta/2, numStates)/2,(1+math.sqrt(2))*math.sqrt(math.log(2/delta)/N_total))

    ite=0

    while delta_w>0 and ite<100:
        recipient_states = [i for i in range(numStates) if P_tilda_sprime[i]<1]
        positive_states = [i for i in range(numStates) if P_tilda_sprime[i]>0]
        ### donor state
        donor_s = min(positive_states, key=lambda x: Vupper[x])
        recipient_s = max(recipient_states, key=lambda x: Vupper[x])
        zeta = min(1-P_tilda_sprime[recipient_s], P_tilda_sprime[donor_s], delta_w)
        if(not P_tilda_sprime[donor_s]>sys.float_info.epsilon):
            break   
        P_tilda_sprime[donor_s] -= zeta
        P_tilda_sprime[recipient_s] += zeta

        delta_w -= zeta
        ite = ite + 1
    # print "ite is ", ite

    return P_tilda_sprime 


def CalculateDelDelV(
    state,
    action,
    mdp,
    N_s_a_sprime,
    Qupper,
    Qlower,
    Vupper,
    Vlower,
    start_state,
    P,
    P_tilda,
    P_lower_tilda,
    R_s_a,
    epsilon,
    delta,
    converge_iterations,
    epsilon_convergence,
    policy_ouu = None
    ):

    Rmax = mdp.Vmax*(1-mdp.discountFactor)
    deldelQ = -1
    ### State has not been observed
    if(np.sum(N_s_a_sprime[state][action])==0):
        deldelQ = mdp.Vmax - mdp.discountFactor*Rmax/(1-mdp.discountFactor)

    else:
        if(policy_ouu is not None):
            P_tilda[state] = UpperP(state,action,delta,N_s_a_sprime[state][action],mdp.numStates,Qupper,False,"ddv")
            P_lower_tilda[state] = LowerP(state,action,delta,N_s_a_sprime[state][action],mdp.numStates,Qupper,False,"ddv")
            Quppernew = itConvergencePolicy(
                Qupper,
                getRewards(R_s_a,policy_ouu),
                P_tilda,
                mdp.discountFactor, 
                epsilon,
                converge_iterations,
                epsilon_convergence
                )
            Qlowernew = itConvergencePolicy(
                Qlower,
                getRewards(R_s_a,policy_ouu),
                P_lower_tilda,
                mdp.discountFactor,
                epsilon,
                converge_iterations,
                epsilon_convergence
                )
            deldelQ = abs(Quppernew[state]-Qlowernew[state]+Qupper[state]-Qlower[state])
        else:
            #### Calculate using changed w
            # print N_s_a_sprime[state][action]
            P_tilda[state][action] = UpperP(state,action,delta,N_s_a_sprime[state][action],mdp.numStates,Vupper,False,"ddv")
            P_lower_tilda[state][action] = LowerP(state,action,delta,N_s_a_sprime[state][action],mdp.numStates,Vupper,False,"ddv")
            Quppernew, Vuppernew = iteratedConvergence(Qupper,R_s_a,P_tilda,mdp.discountFactor, epsilon, converge_iterations, epsilon_convergence)
            Qlowernew, Vlowernew = iteratedConvergence(Qlower,R_s_a,P_lower_tilda,mdp.discountFactor, epsilon, converge_iterations, epsilon_convergence)
            deldelQ = abs(Quppernew[state][action]-Qlowernew[state][action]+Qupper[state][action]-Qlower[state][action])

    if policy_ouu is None:
        policy_ouu = np.argmax(Qupper, axis=1)
    occupancies = np.ones((mdp.numStates))

    #**********************************************************************
    # #### Get occupancy measures 
    # mu_s = [0.0 for i in range(mdp.numStates)]
    # prob = LpProblem("Occupancy solver",LpMinimize)
    # for i in range(mdp.numStates):
    #   mu_s[i] = LpVariable("Occupancy "+str(i), lowBound = 0 ,upBound = mdp.discountFactor/(1-mdp.discountFactor))


    # prob += 1, "Dummy objective function"

    # for st in range(mdp.numStates):
    #   prob += mu_s[st] < mdp.discountFactor/(1-mdp.discountFactor)
    #   prob += mu_s[st] > 0
    #   if(st==start_state):
    #       # rhs = lpSum(transitions[st][ac][sprime]*(rewards[st][ac][sprime]+gamma*V_s[sprime]) for sprime in range(numStates))
    #       prob += mu_s[st] == 1 + mdp.discountFactor*lpSum([mu_s[sprime]*P[sprime][policy_ouu[sprime]][st] for sprime in range(mdp.numStates)])
    #       # prob += V_s[st] >= sum(t*(r+2*v) for t,r,v,g in zip(transitions[st][ac],rewards[st][ac],V_s,gammaList))
    #   else:
    #       prob += mu_s[st] == mdp.discountFactor*lpSum([mu_s[sprime]*P[sprime][policy_ouu[sprime]][st] for sprime in range(mdp.numStates)])
    # prob.writeLP("MDPmodel.lp")
    # prob.solve()
    # # print "occupancy solved"
    #**********************************************************************

    # print "P", P
    for j in range(converge_iterations):
        oldOccupancies = np.copy(occupancies)
        for st in range(mdp.numStates):
            if (st==0):
                occupancies[st] = 1 + mdp.discountFactor*np.sum(occupancies[sprime]*P[sprime][policy_ouu[sprime]][st] for sprime in range(mdp.numStates))
            else:
                occupancies[st] = mdp.discountFactor*np.sum(occupancies[sprime]*P[sprime][policy_ouu[sprime]][st] for sprime in range(mdp.numStates))
        # print oldOccupancies, occupancies
        if(np.linalg.norm(oldOccupancies-occupancies)<=epsilon_convergence):
            break
    # print P[start_state][policy_ouu[start_state]] 
    # print "occupancies", occupancies

    # print state, action, occupancies[state], deldelQ, occupancies[state]*deldelQ
    return occupancies[state]*deldelQ


def getPolicies(numStates, numActions):

    if(numStates==1):
        return range(numActions)

    answer = []
    prev_list = getPolicies(numStates-1, numActions)

    for ac in range(numActions):
        answer += list(map(lambda x: [ac] + x if isinstance(x, (list,)) else [ac,x], prev_list))

    return answer

def getRewards(R_s_a, policy):

    return np.array([R_s_a[x][policy[x]] for x in range(R_s_a.shape[0])])

def getAverageRewards(numStates, numActions, R_s_a_sprime, P_s_a_sprime):
    R_s_a = np.zeros((numStates, numActions))

    for s in range(numStates):
        for a in range(numActions):
            R_s_a[s][a] = np.sum(R_s_a_sprime[s][a] * P_s_a_sprime[s][a]) 

    return R_s_a

def getProb(P_s_a_sprime, policy):

    return np.array([P_s_a_sprime[x,policy[x],:] for x in range(P_s_a_sprime.shape[0])])

def allOneNeighbours(current_policy, numActions):

    answer = []
    for st in range(len(current_policy)):
        for ac in range(numActions):
            if(ac==current_policy[st]):
                continue
            else:
                temp = np.copy(current_policy)
                temp[st] = ac
                answer.append(temp)

    return answer

def indexOfPolicy(policy, numStates, numActions):

    answer = 0

    for i in range(len(policy)):
        answer += policy[i]*(numActions**(numStates-i-1))

    return answer


def getBestPairFromDDV(policy, mdp):
    deltadeltaV = np.zeros((mdp.numStates))
    for st in range(mdp.numStates):
        ac = policy[st]
        #### Compute del del V
        deltadeltaV[st] = CalculateDelDelV(
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
            epsilon_convergence,
            policy1Index
            )



def QSolver (mdp, P, Qinit, stop) :
    """
    Iterate the Bellman Optimality Equations
    to solve for the Q-function. 

    Parameters
    ----------
    mdp : MDP
        MDP object with rewards, discount factor
        and other relevant information.
    P : np.ndarray
        Estimates of the Transition Probabilities.
    Qinit : np.ndarray
        Initial estimates of Q-function.
    stop : lambda
        A function which takes the iteration
        count and difference between
        successive Q-functions and decides
        whether to stop or not.
    """
    iterCnt = 0
    error = math.inf
    Q = np.copy(Qinit)
    while not stop(iterCnt, error) :
        Qold = np.copy(Q)
        V = np.max(Q, axis=1)
        Q = mdp.rewards + mdp.discountFactor * np.sum (P * V, axis=2)
        iterCnt += 1
        error = np.linalg.norm(Q - Qold)
    return Q

RNG = np.random.RandomState(0)

def argmax(a) : 
    """
    If there is a tie between multiple
    entries, we want to ensure that
    different entries are chosen.

    Parameters
    ----------
    a : np.ndarray
        1D array.
    """
    maxs = a == np.max(a)
    nTrue = np.sum(maxs)
    probs = maxs / nTrue
    return RNG.choice(range(a.size), p=probs)

def argmin(a) : 
    """
    If there is a tie between multiple
    entries, we want to ensure that
    different entries are chosen.

    Parameters
    ----------
    a : np.ndarray
        1D Array.
    """
    mins = a == np.min(a)
    nTrue = np.sum(mins)
    probs = mins / nTrue
    return RNG.choice(range(a.size), p=probs)

def confidenceRadius (mdp, visitCount, delta) :
    """
    Referred to as omega in the DDV paper.
    Some magic function probably used to
    make the PAC guarantees go through.

    Parameters
    ----------
    mdp : MDP
        Underlying MDP.
    visitCount : int
        Number of visits to a particular
        (state, action) pair for which
        we are calculating the radius.
    delta : float
        A confidence interval parameter.
    """
    top = np.log(2 ** mdp.numStates - 2) - np.log(delta)
    return np.sqrt(2 * top / (visitCount))

def QBoundsSolver (mdp, PHat, Qinit, N, delta, sense, stop) :
    """
    Solve the Bellman Equations (7) and (8) 
    from the DDV paper:

    PAC Optimal MDP Planning with Application to 
    Invasive Species Management. (Taleghan et al. - 2013)

    Parameters
    ----------
    mdp : MDP
        Underlying MDP.
    PHat : np.ndarray
        Estimate of the MDP's 
        transition probabilities.
    Qinit : np.ndarray
        Initial guess of action-
        value function.
    N : np.ndarray
        State-Action visit count.    
    delta : float
        Confidence Interval Parameter
    sense : bool
        If true, then we solve
        to find the upper bound.
        Else the lower bound.
    stop : lambda
        The stopping condition.
    """
    
    def shiftP (Q, s, a, omega) :
        """
        Helper function used to do value
        iteration with confidence intervals.

        This function gives the probability 
        distribution in the confidence interval
        of the transition probability function
        that will maximize/minimize the outer 
        max/min in the Bellman Equation.

        Based on the procedure described in :

        An Analysis of model-based Interval Estimation
        for Markov Decision Processes. (Strehl, Littman - 2008)
        
        Parameters
        ----------
        Q : np.ndarray
            Action-Value function
        s : int
            State.
        a : int 
            Action.
        omega : float
            Confidence Interval width.
        """
        V = np.max(Q, axis=1)
        Pt = np.copy(PHat[s, a])

        addSelect = argmax if sense else argmin
        subSelect = argmin if sense else argmax
        
        val1 = -math.inf if sense else math.inf
        val2 = math.inf if sense else -math.inf

        # First add amount omega
        # to all the promising states.
        addAmount = omega
        while addAmount > 1e-5 : 
            V1 = np.copy(V)
            mask = Pt < 1
            V1[~mask] = val1
            s = addSelect(V1)
            zeta = min(1 - Pt[s], addAmount)
            Pt[s] += zeta
            addAmount -= zeta

        # Then subtract that value
        # from the less promising states.
        subAmount = omega
        while subAmount > 1e-5 :
            V1 = np.copy(V)
            mask = Pt > 0
            V1[~mask] = val2
            s = subSelect(V1)
            zeta = min(Pt[s], subAmount)
            Pt[s] -= zeta
            subAmount -= zeta

        return Pt / np.sum(Pt)

    iterCnt = 0
    error = math.inf
    Q = np.copy(Qinit)
    while not stop(iterCnt, error) :
        Qold = np.copy(Q)
        Pt = np.zeros(PHat.shape)
        for s, a in product(range(mdp.numStates), range(mdp.numActions)) :
            omega = confidenceRadius(mdp, N[s, a], delta)/2
            Pt[s, a] = shiftP(Q, s, a, omega)
        Q = QSolver(mdp, Pt, Q, stop)
        error = np.linalg.norm(Q - Qold)
    return Q


def QPIBoundsSolver (mdp, pi, PHat, Qinit, N, delta, sense, stop) :
    """
    Solve the Bellman Equations (7) and (8) 
    from the DDV paper:

    PAC Optimal MDP Planning with Application to 
    Invasive Species Management. (Taleghan et al. - 2013)

    Parameters
    ----------
    mdp : MDP
        Underlying MDP.
    PHat : np.ndarray
        Estimate of the MDP's 
        transition probabilities.
    Qinit : np.ndarray
        Initial guess of action-
        value function.
    N : np.ndarray
        State-Action visit count.    
    delta : float
        Confidence Interval Parameter
    sense : bool
        If true, then we solve
        to find the upper bound.
        Else the lower bound.
    stop : lambda
        The stopping condition.
    """
    
    def shiftP (Q, s, a, omega) :
        """
        Helper function used to do value
        iteration with confidence intervals.

        This function gives the probability 
        distribution in the confidence interval
        of the transition probability function
        that will maximize/minimize the outer 
        max/min in the Bellman Equation.

        Based on the procedure described in :

        An Analysis of model-based Interval Estimation
        for Markov Decision Processes. (Strehl, Littman - 2008)
        
        Parameters
        ----------
        Q : np.ndarray
            Action-Value function
        s : int
            State.
        a : int 
            Action.
        omega : float
            Confidence Interval width.
        """
        #import pdb
        #pdb.set_trace()
        V = Q[np.arange(mdp.numStates), pi]
        Pt = np.copy(PHat[s, a])

        addSelect = argmax if sense else argmin
        subSelect = argmin if sense else argmax
        
        val1 = -math.inf if sense else math.inf
        val2 = math.inf if sense else -math.inf

        # First add amount omega
        # to all the promising states.
        addAmount = omega
        while addAmount > 1e-5 : 
            V1 = np.copy(V)
            mask = Pt < 1
            V1[~mask] = val1
            s = addSelect(V1)
            zeta = min(1 - Pt[s], addAmount)
            Pt[s] += zeta
            addAmount -= zeta

        # Then subtract that value
        # from the less promising states.
        subAmount = omega
        while subAmount > 1e-5 :
            V1 = np.copy(V)
            mask = Pt > 0
            V1[~mask] = val2
            s = subSelect(V1)
            zeta = min(Pt[s], subAmount)
            Pt[s] -= zeta
            subAmount -= zeta

        return Pt / np.sum(Pt)

    iterCnt = 0
    error = math.inf
    Q = np.copy(Qinit)
    while not stop(iterCnt, error) :
        iterCnt += 1
        Qold = np.copy(Q)
        Pt = np.zeros(PHat.shape)
        for s, a in product(range(mdp.numStates), range(mdp.numActions)) :
            omega = confidenceRadius(mdp, N[s, a], delta) / 2
            Pt[s, a] = shiftP(Q, s, a, omega)
        Q = QSolver(mdp, Pt, Q, stop)
        error = np.linalg.norm(Q - Qold)
    return Q[np.arange(mdp.numStates), pi]





