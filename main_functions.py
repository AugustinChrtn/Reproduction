"""Contains the functions to run the simulations and the visualisation functions."""
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import time
import itertools
from multiprocessing import Pool

from lopesworld import Lopes_environment

from policy_functions import value_iteration, get_optimal_policy
from policy_functions import policy_evaluation, get_agent_current_policy
from agents import Rmax, BEB, EpsilonMB, EBLP, RmaxLP


def loading_environments():
    """Load all the environment previously generated by the 'generation_env.py' file."""
    env_param = {}
    malus_0_1 = np.load('Environments/Rewards_-0.1.npy', allow_pickle=True)
    transitions_0_1 = np.load('Environments/Transitions_-0.1_1.npy', allow_pickle=True)
    env_param['Lopes'] = {'transitions': transitions_0_1, 'rewards': malus_0_1}

    for index_non_stat in range(1, 21):

        transitions_non_stat_article = np.load(
            'Environments/Transitions_non_stat_article_-0.1_1_' + str(index_non_stat) + '.npy',
            allow_pickle=True)

        transitions_strong_non_stat = np.load(
            'Environments/Transitions_strong_non_stat_-0.1_1_'+str(index_non_stat)+'.npy',
            allow_pickle=True)

        env_param["Lopes_non_stat_article_{0}".format(index_non_stat)] = {
            'transitions': transitions_0_1, 'rewards': malus_0_1,
            'transitions_after_change': transitions_non_stat_article}

        env_param["Lopes_strong_non_stat_{0}".format(index_non_stat)] = {
            'transitions': transitions_0_1, 'rewards': malus_0_1,
            'transitions_after_change': transitions_strong_non_stat}

    malus_1 = np.load('Environments/Rewards_-1.npy', allow_pickle=True)
    for nb_env in range(1, 11):
        transitions_1 = np.load('Environments/Transitions_-1_' + str(nb_env) + '.npy',
                                allow_pickle=True)
        env_param['Stationary_-1_' + str(nb_env)] = {
            'transitions': transitions_1, 'rewards': malus_1}
        for index_non_stat in range(1, 11):

            transitions_non_stat_article = np.load(
                'Environments/Transitions_non_stat_article_-1_' +
                str(nb_env)+'_'+str(index_non_stat)+'.npy',
                allow_pickle=True)

            transitions_strong_non_stat = np.load(
                'Environments/Transitions_strong_non_stat_-1_' +
                str(nb_env)+'_'+str(index_non_stat)+'.npy',
                allow_pickle=True)

            env_param["Non_stat_article_-1_{0}".format(nb_env)+'_{0}'.format(index_non_stat)] = {
                'transitions': transitions_0_1, 'rewards': malus_1,
                'transitions_after_change': transitions_non_stat_article}
            env_param["Non_stat_strong_-1_{0}".format(nb_env)+'_{0}'.format(index_non_stat)] = {
                'transitions': transitions_0_1, 'rewards': malus_1,
                'transitions_after_change': transitions_strong_non_stat}

    return env_param


def getting_simulations_to_do(*args):
    return [i for i in itertools.product(*args)]


def play(environment, name_agent, agent_parameters, trials=100, max_step=30, screen=False,
         photos=[10, 20, 50, 80, 99], accuracy_VI=1e-3, step_between_VI=50):
    """
    Run the agents in the list :name_agent: on all the environment in the list :environment:.

    Parameters
    ----------
    environment : list of Lopes_environment
    name_agent : list of str
        The str are the names of the agents to simulate.
    agent_parameters : dict
        The keys are the names of the agents and the values are dictionaries with their parameters.
    trials : int
    max_step : int
    screen : bool
        If True, saves 2D representation of the Q-values of the agents in the folder Images.
    photos : list of int
        If screen is True, trial at which to take a picture of the Q-values of the agents.
    accuracy_VI : float (>0)
        Threshold after which the value iteration stops updating Q-values.
    step_between_VI : int
        Number of steps between each value iteration.

    Returns
    -------
    reward_per_episode : list
        Rewards accumulated for each trial.
    optimal_policy_value_error : list
        Policy value error for the initial state between the OPTIMAL policy of the agent and the
        best policy that the agent can find.
    real_policy_value_error : list
        Policy value error for the initial state between the REAL policy of the agent and the
        best policy that the agent can find.

    """
    agents = {'R-max': Rmax, 'ζ-R-max': RmaxLP, 'BEB': BEB, 'ζ-EB': EBLP, 'ε-greedy': EpsilonMB}
    agent = agents[name_agent](environment, **agent_parameters)

    reward_per_episode, optimal_policy_value_error, real_policy_value_error = [], [], []
    val_iteration, _ = value_iteration(environment, agent.gamma, accuracy_VI)

    for trial in range(trials):
        if screen:
            take_picture(agent, trial, environment, photos)
        cumulative_reward, step, game_over = 0, 0, False
        while not game_over:
            if environment.total_steps == 900:
                val_iteration, _ = value_iteration(environment, agent.gamma, accuracy_VI)
            if agent.step_counter % step_between_VI == 0:

                best_policy_value_s0 = val_iteration[environment.first_location]

                optimal_policy = get_optimal_policy(agent, agent.gamma, accuracy_VI)
                optimal_policy_value = policy_evaluation(
                    environment, optimal_policy, agent.gamma, accuracy_VI)
                optimal_policy_value_s0 = optimal_policy_value[environment.first_location]
                optimal_policy_value_error.append(optimal_policy_value_s0 - best_policy_value_s0)

                real_policy = get_agent_current_policy(agent)
                real_policy_value = policy_evaluation(
                    environment, real_policy, agent.gamma, accuracy_VI)
                real_policy_value_s0 = real_policy_value[environment.first_location]

                real_policy_value_error.append(real_policy_value_s0-best_policy_value_s0)

            old_state = environment.current_location
            action = agent.choose_action()
            reward, new_state = environment.make_step(action)
            agent.learn(old_state, reward, new_state, action)
            cumulative_reward += reward
            step += 1
            if step == max_step:
                game_over = True
                environment.current_location = environment.first_location
        reward_per_episode.append(cumulative_reward)
    return reward_per_episode, optimal_policy_value_error, real_policy_value_error


def one_parameter_play_function(all_params):
    return play_with_params(all_params[0], all_params[1], all_params[2], all_params[3])


def main_function(all_seeds, every_simulation, play_params, agent_parameters):
    """Run many play functions in parallel using multiprocessing"""
    before = time.time()
    if type(agent_parameters) == dict:
        all_parameters = [[play_params, all_seeds[index_seed], agent_parameters,
                           every_simulation[index_seed]] for index_seed in range(len(all_seeds))]
    else:
        all_parameters = [[play_params, all_seeds[index_seed], agent_parameters[index_seed],
                           every_simulation[index_seed]] for index_seed in range(len(all_seeds))]
    pool = Pool()
    results = pool.map(one_parameter_play_function, all_parameters)
    pool.close()
    pool.join()
    opt_pol_errors, rewards, real_pol_errors = {}, {}, {}
    for result in results:
        opt_pol_errors[result[0]] = result[1][1]
        rewards[result[0]] = result[1][0]
        real_pol_errors[result[0]] = result[1][2]
    time_after = time.time()
    print('Computation time: '+str(time_after - before))
    return opt_pol_errors, real_pol_errors, rewards


def play_with_params(play_parameters, seed, agent_parameters, simulation_to_do):
    np.random.seed(seed)
    environment_parameters = loading_environments()
    name_environment, name_agent, iteration = simulation_to_do[:3]
    environment = Lopes_environment(**environment_parameters[name_environment])
    return simulation_to_do, play(environment, name_agent,
                                  agent_parameters[name_agent], **play_parameters)


def get_mean_and_SEM(dictionary, name_env, agents_tested, nb_iters):
    """Compute the mean and the standard error of the mean of a dictionnary of results."""
    mean = {name_agent: np.average([dictionary[env, name_agent, i]
                                    for i in range(nb_iters) for env in name_env], axis=0)
            for name_agent in agents_tested}
    SEM = {name_agent: (np.std([dictionary[env, name_agent, i]
                               for env in name_env for i in range(nb_iters)], axis=0)
           / np.sqrt(nb_iters*len(name_env))) for name_agent in agents_tested}

    return (mean, SEM)


def extracting_results(rewards, opt_pol_error, real_pol_error, names_environments,
                       agents_tested, number_of_iterations):
    """Apply the function 'get_mean_and_SEM' to all the results of the simulations."""
    optimal_policy_statistics = get_mean_and_SEM(opt_pol_error, names_environments,
                                                 agents_tested, number_of_iterations)
    real_policy_statistics = get_mean_and_SEM(real_pol_error, names_environments,
                                              agents_tested, number_of_iterations)
    rewards_statistics = get_mean_and_SEM(rewards, names_environments,
                                          agents_tested, number_of_iterations)

    return optimal_policy_statistics, real_policy_statistics, rewards_statistics


def evaluate_agents(environments, agents, nb_iters, play_params, agent_parameters, starting_seed):
    """Launch the experiment, extract the results and plot them."""
    every_simulation = getting_simulations_to_do(environments, agents, range(nb_iters))
    all_seeds = [starting_seed+i for i in range(len(every_simulation))]
    pol_errors_opti, pol_errors_real, rewards = main_function(
        all_seeds, every_simulation, play_params, agent_parameters)
    optimal, real, rewards = extracting_results(
        rewards, pol_errors_opti, pol_errors_real, environments, agents, nb_iters)
    save_and_plot(optimal, real, rewards, agents, environments,
                  play_params, environments, agent_parameters)


# Basic visualisation #

def save_and_plot(optimal_stats, real_stats, rewards_stats, agents_tested,
                  names_environments, play_parameters, environment_parameters, agent_parameters):
    """Save and plot the graphical reprentations of the results."""
    pol_opti, SEM_pol_opti = optimal_stats
    pol_real, SEM_pol_real = real_stats
    reward, SEM_reward = rewards_stats
    time_end = str(round(time.time() % 1e7))

    np.save('Results/'+time_end+'_polerror_opti.npy', pol_opti)
    np.save('Results/'+time_end+'_polerror_real.npy', pol_real)
    np.save('Results/'+time_end+'_rewards.npy', reward)

    colors = {'R-max': '#9d02d7', 'ζ-R-max': '#0000ff',
              'ε-greedy': "#ff7763", 'BEB': "#ffac1e", 'ζ-EB': "#009435"}
    marker_sizes = {'R-max': '2', 'ζ-R-max': '2', 'BEB': '2', 'ζ-EB': '2', 'ε-greedy': '2'}
    markers = {'R-max': '^', 'ζ-R-max': 'o', 'BEB': 'x', 'ζ-EB': '*', 'ε-greedy': 's'}

    length_pol = (play_parameters["trials"]*play_parameters["max_step"]
                  )//play_parameters["step_between_VI"]

    plot1D([-13, 0.5], "Steps", "Policy value error")

    x_axis = [i*play_parameters["step_between_VI"] for i in range(length_pol)]
    plot_agents(agents_tested, pol_opti, SEM_pol_opti, x_axis, colors, markers, marker_sizes)
    plt.savefig('Results/pol_error_opti'+time_end +
                names_environments[0]+'.pdf', bbox_inches='tight')

    plot1D([-13, 0.5], "Steps", "Policy value error")
    plot_agents(agents_tested, pol_real, SEM_pol_real, x_axis, colors, markers, marker_sizes)
    plt.savefig('Results/pol_error_real'+time_end +
                names_environments[0]+'.pdf', bbox_inches='tight')
    plt.close()

    x_reward = [i for i in range(play_parameters["trials"])]
    plot1D([-1, 26], "Trials", "Reward")
    plot_agents(agents_tested, reward, SEM_reward, x_reward, colors, markers, marker_sizes)
    plt.savefig('Results/Rewards'+time_end+names_environments[0]+'.pdf', bbox_inches='tight')
    plt.close()


def plot1D(ylim, xlabel, ylabel):
    """Create a basic template for a one dimensional plot."""
    fig = plt.figure(dpi=300)
    fig.add_subplot(1, 1, 1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(linestyle='--')
    plt.ylim(ylim)


def plot_agents(agents_tested, values, SEM_values, x_range, colors, markers, marker_sizes):
    """Plot the performance of all the agents."""
    for name_agent in agents_tested:
        yerr0 = values[name_agent] - SEM_values[name_agent]
        yerr1 = values[name_agent] + SEM_values[name_agent]

        plt.fill_between(x_range, yerr0, yerr1, color=colors[name_agent], alpha=0.2)

        plt.plot(x_range, values[name_agent], color=colors[name_agent], label=name_agent,
                 ms=marker_sizes[name_agent], marker=markers[name_agent])
    plt.legend()

# PICTURES #


def plot_V(table, policy_table, position):
    """Plot the heatmap of maximal Q-values."""
    plt.subplot(position, aspect='equal')
    table = np.reshape(table, (5, 5))
    policy_table = np.reshape(policy_table, (5, 5))
    sns.heatmap(table, cmap='crest', cbar=False, annot=table, fmt='.1f',
                annot_kws={"size": 35 / (np.sqrt(len(table))+2.5)})
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            if policy_table[i, j] == 4:
                circle = plt.Circle((j+0.5, i+0.8), radius=0.07, color='black', fill=True)
                plt.gca().add_patch(circle)
            else:
                rotation = {0: (0.5, 0.9, 0, -0.05), 1: (0.5, 0.65, 0, 0.05),
                            2: (0.65, 0.8, -0.05, 0), 3: (0.4, 0.8, +0.05, 0)}
                rotation_to_make = rotation[policy_table[i, j]]
                plt.arrow(j+rotation_to_make[0], i+rotation_to_make[1], rotation_to_make[2],
                          rotation_to_make[3], head_width=0.12, head_length=0.12, fc='black',
                          ec='black', linewidth=0.5, length_includes_head=False, shape='full',
                          overhang=0, head_starts_at_zero=True)
    plt.xticks([])
    plt.yticks([])


def take_picture(agent, trial, environment, photos):
    """Take a picture of the heatmap of the Q-values of an agent."""
    if trial in photos:
        best_q_values, policies = get_max_Q_values_and_policy(agent.Q)
        plt.figure(dpi=300)
        if type(agent).__name__ == 'EpsilonMB':
            plot_V(best_q_values, policies, 111)
        if type(agent).__name__ in ['BEB', 'Rmax', 'EBLP', 'RmaxLP']:
            if type(agent).__name__ in ['Rmax', 'RmaxLP']:
                bonus = agent.R_VI
            else:
                bonus = agent.bonus
            plot_V(best_q_values, policies, 121)
            best_values_bonus, best_policy_bonus = get_max_Q_values_and_policy(bonus)
            plot_V(best_values_bonus, best_policy_bonus, 122)
        plt.savefig("Images/"+type(agent).__name__+"_"+type(environment).__name__ +
                    "_"+str(trial)+".pdf", bbox_inches='tight')
        plt.close()


def get_max_Q_values_and_policy(table):
    """Receive a table of Q-values and return the max Q-values and the associated actions."""
    best_values = np.max(table, axis=1)
    best_actions = np.argmax(table+1e-5*np.random.random(table.shape), axis=1)
    return best_values, best_actions


def compute_optimal_policies():
    """Generate all the heatmaps of the optimal policies on all the environments."""
    env_param = loading_environments()
    for name_env in env_param.keys():
        if 'transitions_after_change' in env_param[name_env].keys():
            env_param[name_env]['transitions'] = env_param[name_env]['transitions_after_change']
        environment = Lopes_environment(**env_param[name_env])
        V, policy = value_iteration(environment, gamma=0.95, accuracy=1e-3)
        plt.figure(dpi=300)
        plot_V(V, policy, 111)
        plt.savefig("Images/Optimal policy in each environment/VI_" +
                    name_env+".pdf", bbox_inches='tight')
        plt.close()