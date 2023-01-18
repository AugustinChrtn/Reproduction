from main_Functions import evaluate_agents


### Experiments ###


#agents=['R-max','ζ-R-max','BEB','ζ-EB','ε-greedy']
agents=['R-max']
#environments=["Non_stat_article_-1_{0}".format(world)+'_{0}'.format(non_stat) for world in range(1,11) for non_stat in range(1,11), "Non_stat_article_-0.1_{0}".format(non_stat)for non_stat in range(1,6)]

### Reproduction of the figures of Lopes et al. (2012) ###


play_parameters={'trials':100, 'max_step':30, 'screen':0,'photos':[1,2,3,10,20,30,40,50,60,70,80,99],'accuracy_VI':0.01,'step_between_VI':50}


agent_parameters={'ε-greedy':{'gamma':0.95,'epsilon':0.3},
            'R-max':{'gamma':0.95, 'm':8,'Rmax':1,'m_uncertain_states':12,'condition':'informative'},
            'BEB':{'gamma':0.95,'beta':7,'coeff_prior':2,'condition':'informative'},
            'ζ-EB':{'gamma':0.95,'beta':7,'step_update':10,'alpha':1,'prior_LP':0.01},
            'ζ-R-max':{'gamma':0.95,'Rmax':1,'m':2,'step_update':10,'alpha':1,'prior_LP':0.01}}

environments=['Lopes']
nb_iters=20
starting_seed=100
evaluate_agents(environments,agents,nb_iters,play_parameters,agent_parameters,starting_seed)


#Reproduction of the second figure of Lopes et al. (2012)

starting_seed=200
agent_parameters['R-max']['condition']='wrong_prior'
agent_parameters['BEB']['condition']='wrong_prior'

evaluate_agents(environments,agents,nb_iters,play_parameters,agent_parameters,starting_seed)

"""

#Reproduction of the third figure of Lopes et al. (2012)
starting_seed=300
nb_iters=5
environments=["Non_stat_article_-0.1_{0}".format(non_stat)for non_stat in range(1,21)]

agent_parameters['R-max']['condition']='informative'
agent_parameters['BEB']['condition']='informative'

evaluate_agents(environments,agents,nb_iters,play_parameters,agent_parameters,starting_seed)

#Stronger non-stationarity to find the same third figure as Lopes et al. (2012)

starting_seed=400
environments=["Non_stat_strong_-0.1_{0}".format(non_stat)for non_stat in range(1,21)]

evaluate_agents(environments,agents,nb_iters,play_parameters,agent_parameters,starting_seed)


#Without informative prior

environments=['Lopes']
nb_iters=20
starting_seed=250

agent_parameters['R-max']={'gamma':0.95, 'm':10,'Rmax':1,'m_uncertain_states':10,'condition':'uninformative'}
agent_parameters['BEB']={'gamma':0.95,'beta':7,'coeff_prior':0.001,'condition':'uninformative'}

evaluate_agents(environments,agents,nb_iters,play_parameters,agent_parameters,starting_seed)


#With the parameters to maximize the real agent policy

agent_parameters={'ε-greedy':{'gamma':0.95,'epsilon':0.01},
            'R-max':{'gamma':0.95, 'm':8,'Rmax':1,'m_uncertain_states':12,'condition':'informative'},
            'BEB':{'gamma':0.95,'beta':3,'coeff_prior':2,'condition':'informative'},
            'ζ-EB':{'gamma':0.95,'beta':0.5,'step_update':10,'alpha':2.5,'prior_LP':0.01},
            'ζ-R-max':{'gamma':0.95,'Rmax':1,'m':2,'step_update':10,'alpha':1,'prior_LP':0.01}}

environments=["Lopes"]
nb_iters=20

starting_seed=500
evaluate_agents(environments,agents,nb_iters,play_parameters,agent_parameters,starting_seed)

starting_seed=600
agent_parameters['R-max']['condition']='wrong_prior'
agent_parameters['BEB']['condition']='wrong_prior'

evaluate_agents(environments,agents,nb_iters,play_parameters,agent_parameters,starting_seed)

#Reproduction of the third figure of Lopes et al. (2012)
starting_seed=300
nb_iters=5
agent_parameters['R-max']['condition']='informative'
agent_parameters['BEB']['condition']='informative'

environments=["Non_stat_article_-0.1_{0}".format(non_stat)for non_stat in range(1,21)]

evaluate_agents(environments,agents,nb_iters,play_parameters,agent_parameters,starting_seed)

#Stronger non-stationarity to find the same third figure as Lopes et al. (2012)
starting_seed=400
environments=["Non_stat_strong_-0.1_{0}".format(non_stat)for non_stat in range(1,21)]

evaluate_agents(environments,agents,nb_iters,play_parameters,agent_parameters,starting_seed)

#Without informative prior
starting_seed=250
environments=["Lopes"]
nb_iters=20

agent_parameters['R-max']={'gamma':0.95, 'm':8,'Rmax':1,'m_uncertain_states':12,'condition':'uninformative'}
agent_parameters['BEB']={'gamma':0.95,'beta':7,'coeff_prior':0.001,'condition':'uninformative'}

evaluate_agents(environments,agents,nb_iters,play_parameters,agent_parameters,starting_seed)

#Replication of the first figure of Lopes et al. (2012) with a negative reward of -1
starting_seed=1000
#Replication of the third figure of Lopes et al. (2012) with a negative reward of -1
starting_seed=1500

#Stronger non-stationarity than Lopes et al. (2012) with a negative reward of -1
starting_seed=2000
"""
