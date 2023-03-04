#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Large-scale Competitive learning-based Salp Swarm for global optimization and solving Constrained mechanical and engineering design problems   
# More details about the algorithm are in [please cite the original paper ] DOI : 
# Mohammed Qaraad , Abdussalam Aljadania and Mostafa A. Elhosseini, "Large-scale Competitive learning-based Salp Swarm for global optimization and solving Constrained mechanical and engineering design problems  "
# Mathematics, 2023


import random
import numpy
import math
import time
import numpy as np
import matplotlib.pyplot as plt

def  getrandlist(m):
    #print(m)
    randlist = numpy.arange(m) 
    templist = numpy.arange(m) 
    #print(templist.shape)
    end = m - 1
    for i in range(0,m):
      templist[i] = i
    for i in range(0,m):
      k = random.randint(0,end)
      randlist[i] = templist[k]
      templist[k] = templist[end]
      end = end - 1
    return randlist
  
def compete(r1,r2,Fitness):
    if(Fitness[r1] < Fitness[r2]):
        return r1
    return r2

def objective_Fun (x):
    return 20+x[0]**2-10.*np.cos(2*3.14159*x[0])+x[1]**2-10*np.cos(2*3.14159*x[1])

def CL_SSA(objf, lb, ub, dim, SearchAgents_no, Max_iter):

    # destination_pos
    Dest_pos = numpy.zeros(dim)
    Dest_score = float("inf")

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    # Initialize the positions of search agents
    v = numpy.zeros((SearchAgents_no, dim))
    Fitness = numpy.full(SearchAgents_no, float("inf"))
    Positions = numpy.zeros((SearchAgents_no, dim))

    for i in range(dim):
        Positions[:, i] = (
            numpy.random.uniform(0, 1, SearchAgents_no) * (ub[i] - lb[i]) + lb[i]
        )

    Convergence_curve = numpy.zeros(Max_iter)

    # Loop counter
    print('CL_SSA is optimizing  "' + objf.__name__ + '"')

    timerStart = time.time()
    startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    # Main loop
    for l in range(0, Max_iter):
        dec = 2 * math.exp(-((4 * l / Max_iter) ** 2))
        for i in range(0, SearchAgents_no):

            # Return back the search agents that go beyond the boundaries of the search space
            for j in range(dim):
                Positions[i, j] = numpy.clip(Positions[i, j], lb[j], ub[j])

            
            Fitness[i] = objf(Positions[i, :])
#             c_fitness = c_double()
#             cost_function(Positions[i, :].ctypes.data_as(POINTER(c_double)), byref(c_fitness), dim, 1, int(objf))
#             Fitness[i] = c_fitness.value
            
            if Fitness[i] < Dest_score:
                Dest_score = Fitness[i]  # Update Dest_Score
                Dest_pos = Positions[i, :].copy()
       
        #generate a random index list in preparation for pairwise competition 
        randlist = getrandlist(SearchAgents_no)
        #calculate the center (mean position) of the swar
        center = Positions.mean(axis=0)
        # Update the Position of search agents
        for i in range(0, math.ceil(SearchAgents_no/2)):
            r1 = randlist[i]
            r2 = randlist[i + math.ceil(SearchAgents_no/2)]
            #new_update 
            if(Fitness[r1] < Fitness[r2]):
                winidx = r1
                loseidx = r2
            else :
                winidx = r1
                loseidx = r2                
           #update winner ssa
            
#             winidx = compete(r1, r2,Fitness)
#             loseidx = r1 + r2 - winidx

            for j in range(0, dim):

                c1 = random.random()
                c2 = random.random()
                c3 = random.random()
                v[loseidx][j] = c1*v[loseidx][j]+ c2*( Positions[winidx][j] - Positions[loseidx][j])+ c3*0.3*(center[j] - Positions[loseidx][j])
                Positions[loseidx][j] = Positions[loseidx][j] + v[loseidx][j]
                
            for j in range(0, dim):
                c2 = random.random()
                c3 = random.random()
                    # Eq. (3.1) in the paper
                if c3 < 0.5:
                    Positions[winidx][j] = Dest_pos[j] + dec * (
                            (ub[j] - lb[j]) * c2 + lb[j]
                        )
                else:
                    Positions[winidx][j] = Dest_pos[j] - dec * (
                            (ub[j] - lb[j]) * c2 + lb[j]
                        )                
        

        Convergence_curve[l] = Dest_score

        if l % 1 == 0:
            print(
                ["At iteration " + str(l) + " the best fitness is " + str(Dest_score)]
            )



    

    return Convergence_curve


Max_iterations=50  # Maximum Number of Iterations
swarm_size = 30 # Number of salps
LB=-10  #lower bound of solution
UB=10   #upper bound of solution
Dim=2 #problem dimensions
NoRuns=100  # Number of runs
ConvergenceCurve=np.zeros((Max_iterations,NoRuns))
for r in range(NoRuns):
    result = CL_SSA(objective_Fun, LB, UB, Dim, swarm_size, Max_iterations)
    ConvergenceCurve[:,r]=result
# Plot the convergence curves of all runs
idx=range(Max_iterations)
fig= plt.figure()

#3-plot
ax=fig.add_subplot(111)
for i in range(NoRuns):
    ax.plot(idx,ConvergenceCurve[:,i])
plt.title('Convergence Curve of the CL_SSA Optimizer', fontsize=12)
plt.ylabel('Fitness')
plt.xlabel('Iterations')
plt.show()

