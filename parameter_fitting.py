"""Generate the heatmaps and the 1D plots for the parameter fitting."""
import numpy as np
import time
from main_functions import loading_environments, getting_simulations_to_do, main_function, plot1D

import matplotlib.pyplot as plt
import seaborn as sns
import copy

from matplotlib.colors import TwoSlopeNorm

plt.switch_backend("Agg")

name_to_save = {'ε-greedy': 'e-greedy',
                'BEB': 'BEB',
                'R-max': 'R-max',
                'ζ-EB': 'zeta-EB',
                'ζ-R-max': 'zeta-R-max'}


def range_parameters_agent(list1, list2):
    list_of_params = []
    for elem1 in range(1, len(list1)):
        for elem2 in range(1, len(list2)):
            list_of_params.append(
                {list1[0]: list1[elem1], list2[0]: list2[elem2]})
    return list_of_params


def get_agent_parameters(name_agent, basic_parameters, list_1, list_2):
    """Generate all the agent parameters to test during parameter fitting."""
    agent_parameters = []
    list_of_new_parameters = range_parameters_agent(list_1, list_2)
    for dic in list_of_new_parameters:
        d = copy.deepcopy(basic_parameters)
        for key, value in dic.items():
            d[name_agent][key] = value
        agent_parameters.append(d)
    return agent_parameters


def get_mean_and_SEM_fitting(dictionary, agent, names_env, nb_iters, first_param, second_param):
    """ Compute the mean and the standard error of the mean for each of the parameter couples."""
    mean = {(agent, p_1, p_2): np.average([dictionary[env, agent, i, p_1, p_2]
                                           for i in range(nb_iters) for env in names_env], axis=0)
            for p_1 in first_param for p_2 in second_param}
    SEM = {(agent, p_1, p_2): (np.std([dictionary[env, agent, i, p_1, p_2]
                                       for env in names_env for i in range(nb_iters)], axis=0)
                               / np.sqrt(nb_iters*len(names_env)))
           for p_1 in first_param for p_2 in second_param}
    return mean, SEM


def extracting_results(opti_pol_error, real_pol_error, agents_tested, names_env,
                       nb_iters, first_param, second_param):
    """Apply get_mean_and_SEM_fitting to the measures of policy value error (real and optimal)."""
    mean_pol_opti, SEM_pol_opti = get_mean_and_SEM_fitting(
        opti_pol_error, names_env, agents_tested, nb_iters, first_param, second_param)
    mean_pol_real, SEM_pol_real = get_mean_and_SEM_fitting(
        real_pol_error, names_env, agents_tested, nb_iters, first_param, second_param)
    return mean_pol_opti, SEM_pol_opti, mean_pol_real, SEM_pol_real


def get_best_performance(pol_error, name_agent, first_param, second_param, range_of_the_mean):
    avg_pol_error = {(name_agent, hp_1, hp_2): np.average(pol_error_value[-range_of_the_mean:])
                     for (name_agent, hp_1, hp_2), pol_error_value in pol_error.items()}

    array_result = np.zeros((len(first_param)-1, len(second_param)-1))
    for index_hp_1, hp_1 in enumerate(first_param[1:]):
        for index_hp_2, hp_2 in enumerate(second_param[1:]):
            array_result[(index_hp_1, index_hp_2)
                         ] = avg_pol_error[name_agent, hp_1, hp_2]
    return array_result


def save_results_parameter_fitting(pol_opti, SEM_pol_opti, pol_real, SEM_pol_real,
                                   name_agent, first_param, second_param,
                                   play_parameters, name_environments):
    """Save all the results of the parameter fitting in a '.npy' format."""
    time_end = str(round(time.time() % 1e7))
    tmp_path = 'Parameter fitting/Data/'+name_to_save[name_agent]+'/'
    path = tmp_path+name_agent+'_' + name_environments[0]
    np.save(path+'_pol_opti_'+time_end+'.npy', pol_opti)
    np.save(path+'_SEM_pol_opti_'+time_end+'.npy', SEM_pol_opti)
    np.save(path+'_pol_real_'+time_end+'.npy', pol_real)
    np.save(path+'_SEM_pol_real_'+time_end+'.npy', SEM_pol_real)
    return time_end


def plot_from_saved(name_agent, first_param, second_param,
                    step_between_VI, name_environments, time_end, optimal):
    path = 'Parameter fitting/Data/'+name_to_save[name_agent]+'/'
    if optimal:
        pol = np.load(path+name_agent+'_' + name_environments[0] +
                      '_pol_opti_'+time_end+'.npy', allow_pickle=True)[()]
        SEM_pol = np.load(path+name_agent+'_' + name_environments[0] +
                          '_SEM_pol_opti_'+time_end+'.npy', allow_pickle=True)[()]
    else:
        pol = np.load(path+name_agent+'_' + name_environments[0] +
                      '_pol_real_'+time_end+'.npy', allow_pickle=True)[()]
        SEM_pol = np.load(path+name_agent+'_' + name_environments[0] +
                          '_SEM_pol_real_'+time_end+'.npy', allow_pickle=True)[()]
    plot_parameter_fitting(pol, SEM_pol, name_agent, first_param,
                           second_param, step_between_VI, name_environments, time_end, optimal)


def plot_parameter_fitting(pol, SEM_pol, name_agent, first_param, second_param,
                           step_between_VI, name_environments, time_end, optimal):
    """Plot 1D curves for parameter fitting."""
    markers = ['^', 'o', 'x', '*', 's']
    colors = ['#9d02d7', '#0000ff', "#ff7763", "#ffac1e", "#009435"]
    array_avg_pol_last_500 = get_best_performance(
        pol, name_agent, first_param, second_param, 10)

    plot_2D(array_avg_pol_last_500, first_param, second_param)
    path = 'Parameter fitting/Heatmaps/'+name_to_save[name_agent]+'/'
    if optimal:
        plt.title(name_agent+' optimal policy - last 500 steps')
        plt.savefig(path+name_agent+' optimal_policy ' +
                    name_environments[0]+time_end+'.pdf', bbox_inches='tight')
    else:
        plt.title(name_agent+' agent policy - last 500 steps')
        plt.savefig(path+name_agent+' real_policy ' +
                    name_environments[0]+time_end+'.pdf', bbox_inches='tight')
    plt.close()

    curve_number = 0
    for hp_1 in first_param[1:]:
        plot1D(ylim=[-12.5, 0.5], xlabel="Steps", ylabel="Policy value error")
        for hp_2 in second_param[1:]:
            yerr0 = pol[name_agent, hp_1, hp_2] - \
                SEM_pol[name_agent, hp_1, hp_2]
            yerr1 = pol[name_agent, hp_1, hp_2] + \
                SEM_pol[name_agent, hp_1, hp_2]

            plt.fill_between([step_between_VI*i for i in range(len(pol[name_agent, hp_1, hp_2]))],
                             yerr0, yerr1, color=colors[curve_number], alpha=0.2)

            plt.plot([step_between_VI*i for i in range(len(pol[name_agent, hp_1, hp_2]))],
                     pol[name_agent, hp_1, hp_2], color=colors[curve_number],
                     label=str(second_param[0])+"="+str(hp_2), ms=4, marker=markers[curve_number])
            curve_number += 1
            if curve_number == 5 or hp_2 == second_param[-1]:
                plt.legend()
                path = 'Parameter fitting/1DPlots/' + \
                    name_to_save[name_agent] + '/'
                if optimal:
                    plt.title(name_agent+" optimal policy with " +
                              str(first_param[0])+" = "+str(hp_1))
                    plt.savefig(path+'pol_error_opti_' + name_agent + '_' +
                                name_environments[0] + str(hp_2)+"_"
                                + str(time.time())+'.pdf', bbox_inches='tight')
                else:
                    plt.title(name_agent+" agent policy with " +
                              str(first_param[0])+" = "+str(hp_1))
                    plt.savefig(path+'pol_error_real_'+name_agent + '_' +
                                name_environments[0] + str(hp_2) + "_" +
                                str(time.time())+'.pdf', bbox_inches='tight')
                plt.close()
                curve_number = 0
                if hp_2 != second_param[-1]:
                    plot1D(ylim=[-12.5, 0.5], xlabel="Steps",
                           ylabel="Policy value error")


def plot_2D(array_result, first_param, second_param):
    """Provide a template for 2D heatmaps."""
    fig = plt.figure(dpi=300)
    fig.add_subplot(1, 1, 1)
    divnorm = TwoSlopeNorm(vmin=-12, vcenter=-1, vmax=0)
    font_size = 35 / (np.sqrt(len(array_result))+2.5)
    sns.heatmap(array_result, cmap='bwr', norm=divnorm, cbar=True, annot=np.round(array_result, 1),
                cbar_kws={"ticks": [-12, -1, 0]}, annot_kws={"size": font_size})
    plt.xlabel(second_param[0])
    plt.ylabel(first_param[0])
    plt.xticks([i+0.5 for i in range(len(second_param[1:]))], second_param[1:])
    plt.yticks([i+0.5 for i in range(len(first_param[1:]))], first_param[1:])


def fit_parameters_agent(environments, name_agent, nb_iters, first_hp, second_hp,
                         basic_param, starting_seed, play_parameters):

    every_simulation = getting_simulations_to_do(
        environments, [name_agent], range(nb_iters), first_hp[1:], second_hp[1:])
    seeds_agent = [starting_seed+i for i in range(len(every_simulation))]
    length_agent_param = nb_iters * len(environments)
    agent_parameters = length_agent_param*get_agent_parameters(name_agent, basic_param,
                                                               first_hp, second_hp)
    opti_pol_errors, real_pol_errors, reward_pol_errors = main_function(
        seeds_agent, every_simulation, play_parameters, agent_parameters)
    mean_pol_opti, SEM_pol_opti, mean_pol_real, SEM_pol_real = extracting_results(
        opti_pol_errors, real_pol_errors, environments,
        name_agent, nb_iters, first_hp[1:], second_hp[1:])
    time_end = save_results_parameter_fitting(
        mean_pol_opti, SEM_pol_opti, mean_pol_real, SEM_pol_real, name_agent,
        first_hp, second_hp, play_parameters, environments)
    for optimal in [False, True]:
        plot_from_saved(name_agent, first_hp, second_hp,
                        play_parameters["step_between_VI"], environments, time_end, optimal)


# Parameter fitting for each agent
if __name__ == "__main__":
    print("""This program launches all the parameter fitting.
          
          There are 19 parameter fits with different agent or parameter ranges.
          
          For each experiment, we print the computational time each fit took.

          This code uses multiprocessing and runs faster on machines with many
          cores. 

          An estimation of the total computational time is 30 times the time
          of the first experiment (e.g 100 min if the first experiment takes 200
          seconds.)

          Parameter fitting can take a few hours to run.

          """)

    environments_parameters = loading_environments()
    play_params = {'trials': 100, 'max_step': 30, 'screen': False, 'photos': [
        10, 20, 50, 80, 99], 'accuracy_VI': 0.01, 'step_between_VI': 50}

    # Reproduction of Lopes et al. (2012)

    environments = ['Lopes']
    nb_iters = 20

    agent_parameters = {
        'ε-greedy': {'gamma': 0.95, 'epsilon': 0.3},
        'R-max': {'gamma': 0.95, 'm': 8, 'Rmax': 1, 'm_u': 12, 'condition': 'informative'},
        'BEB': {'gamma': 0.95, 'beta': 7, 'coeff_prior': 2, 'condition': 'informative'},
        'ζ-EB': {'gamma': 0.95, 'beta': 3, 'step_update': 10, 'alpha': 0.3, 'prior_LP': 0.01},
        'ζ-R-max': {'gamma': 0.95, 'Rmax': 1, 'm': 2, 'step_update': 10, 'alpha': 0.3, 'prior_LP': 0.01}}

    agent_name = 'R-max'
    starting_seed = 10000

    first_hp = ['m']+[1]+[i for i in range(5, 41, 5)]
    second_hp = ['m_u']+[1]+[i for i in range(5, 41, 5)]
    fit_parameters_agent(environments, agent_name, nb_iters, first_hp, second_hp,
                         {agent_name: agent_parameters[agent_name]}, starting_seed, play_params)

    first_hp = ['m']+[i for i in range(2, 15, 2)]
    second_hp = ['m_u']+[i for i in range(2, 21, 2)]
    fit_parameters_agent(environments, agent_name, nb_iters, first_hp, second_hp,
                         {agent_name: agent_parameters[agent_name]}, starting_seed, play_params)

    agent_parameters['R-max']['condition'] = 'uniform'
    first_hp = ['gamma']+[0.95]
    second_hp = ['m']+[i for i in range(2, 21, 2)]
    fit_parameters_agent(environments, agent_name, nb_iters, first_hp, second_hp,
                         {agent_name: agent_parameters[agent_name]}, starting_seed, play_params)

    agent_name = 'ζ-R-max'
    starting_seed = 20000

    first_hp = ['m']+[round(i*0.1, 1) for i in range(5, 41, 5)]
    second_hp = ['alpha']+[0.01, 0.05, 0.1, 0.3, 0.5, 1, 1.5, 2, 3]
    fit_parameters_agent(environments, agent_name, nb_iters, first_hp, second_hp,
                         {agent_name: agent_parameters[agent_name]}, starting_seed, play_params)

    first_hp = ['m']+[round(i*0.1, 1) for i in range(20, 31, 2)]
    second_hp = ['alpha']+[0.01, 0.05, 0.1, 0.3, 0.5, 1, 1.5, 2, 3]
    fit_parameters_agent(environments, agent_name, nb_iters, first_hp, second_hp,
                         {agent_name: agent_parameters[agent_name]}, starting_seed, play_params)

    first_hp = ['gamma']+[0.95]
    second_hp = ['prior_LP']+[10**(i) for i in range(-5, 4)]
    fit_parameters_agent(environments, agent_name, nb_iters, first_hp, second_hp,
                         {agent_name: agent_parameters[agent_name]}, starting_seed, play_params)

    agent_name = 'BEB'
    starting_seed = 30000

    first_hp = ['coeff_prior']+[2]
    second_hp = ['beta']+[0.1]+[1]+[2]+[3]+[4]+[i for i in range(5, 26, 5)]
    fit_parameters_agent(environments, agent_name, nb_iters, first_hp, second_hp,
                         {agent_name: agent_parameters[agent_name]}, starting_seed, play_params)

    starting_seed = 31000
    first_hp = ['beta']+[10]
    second_hp = ['coeff_prior']+[10**i for i in range(-1, 4)]
    fit_parameters_agent(environments, agent_name, nb_iters, first_hp, second_hp,
                         {agent_name: agent_parameters[agent_name]}, starting_seed, play_params)

    agent_parameters['BEB']['condition'] = 'uniform'

    first_hp = ['coeff_prior']+[0.001]
    second_hp = ['beta']+[0.1]+[i for i in range(1, 10)]
    fit_parameters_agent(environments, agent_name, nb_iters, first_hp, second_hp,
                         {agent_name: agent_parameters[agent_name]}, starting_seed, play_params)

    starting_seed = 35000

    agent_parameters['BEB']['condition'] = 'informative'
    first_hp = ['coeff_prior']+[0.1]+[1]+[3]+[i for i in range(5, 26, 5)]
    second_hp = ['beta']+[0.1]+[1]+[2]+[3]+[4]+[i for i in range(5, 26, 5)]
    fit_parameters_agent(environments, agent_name, nb_iters, first_hp, second_hp,
                         {agent_name: agent_parameters[agent_name]}, starting_seed, play_params)

    agent_name = 'ζ-EB'
    starting_seed = 40000

    first_hp = ['beta']+[0.1]+[i for i in range(1, 20, 2)]
    second_hp = ['alpha']+[0.01, 0.05, 0.1, 0.3, 0.5, 1, 1.5, 2, 3]
    fit_parameters_agent(environments, agent_name, nb_iters, first_hp, second_hp,
                         {agent_name: agent_parameters[agent_name]}, starting_seed, play_params)

    first_hp = ['alpha']+[0.01, 0.05, 0.1, 0.3, 0.5, 1, 1.5, 2, 3]
    second_hp = ['prior_LP']+[10**(i) for i in range(-3, 2)]
    fit_parameters_agent(environments, agent_name, nb_iters, first_hp, second_hp, {
        agent_name: agent_parameters[agent_name]}, starting_seed, play_params)

    first_hp = ['beta']+[0.1]+[round(0.1*i, 1) for i in range(2, 11, 2)]
    second_hp = ['alpha']+[0.01, 0.05, 0.1, 0.3, 0.5, 1, 1.5, 2, 3]
    fit_parameters_agent(environments, agent_name, nb_iters, first_hp, second_hp,
                         {agent_name: agent_parameters[agent_name]}, starting_seed, play_params)

    agent_name = 'ε-greedy'
    starting_seed = 50000

    first_hp = ['gamma']+[0.95]
    second_hp = ['epsilon']+[0.001]+[0.01]+[0.05] + \
        [round(i*0.1, 1) for i in range(1, 4, 1)]+[round(i*0.1, 1)
                                                   for i in range(5, 11, 2)]
    fit_parameters_agent(environments, agent_name, nb_iters, first_hp, second_hp,
                         {agent_name: agent_parameters[agent_name]}, starting_seed, play_params)

    # Parameter fitting for the environments with a malus of -1

    environments = ['Stationary_-1_'+str(number_world)
                    for number_world in range(1, 11)]
    nb_iters = 10

    agent_parameters = {
        'ε-greedy': {'gamma': 0.95, 'epsilon': 0.3},
        'R-max': {'gamma': 0.95, 'm': 8, 'Rmax': 1, 'm_u': 12, 'condition': 'informative'},
        'BEB': {'gamma': 0.95, 'beta': 7, 'coeff_prior': 2, 'condition': 'informative'},
        'ζ-EB': {'gamma': 0.95, 'beta': 3, 'step_update': 10, 'alpha': 0.3, 'prior_LP': 0.01},
        'ζ-R-max': {'gamma': 0.95, 'Rmax': 1, 'm': 2, 'step_update': 10, 'alpha': 0.3, 'prior_LP': 0.01}}

    agent_name = 'R-max'
    starting_seed = 60000

    first_hp = ['m']+[1]+[i for i in range(5, 41, 5)]
    second_hp = ['m_u']+[1]+[i for i in range(5, 41, 5)]

    fit_parameters_agent(environments, agent_name, nb_iters, first_hp, second_hp,
                         {agent_name: agent_parameters[agent_name]}, starting_seed, play_params)

    agent_name = 'ζ-R-max'
    starting_seed = 70000

    first_hp = ['m']+[round(i*0.1, 1) for i in range(5, 41, 5)]
    second_hp = ['alpha']+[0.01, 0.05, 0.1, 0.3, 0.5, 1, 1.5, 2, 3]
    fit_parameters_agent(environments, agent_name, nb_iters, first_hp, second_hp,
                         {agent_name: agent_parameters[agent_name]}, starting_seed, play_params)

    agent_name = 'BEB'
    starting_seed = 80000

    first_hp = ['coeff_prior']+[2]
    second_hp = ['beta']+[0.1]+[1]+[2]+[3]+[4]+[i for i in range(5, 26, 5)]
    fit_parameters_agent(environments, agent_name, nb_iters, first_hp, second_hp,
                         {agent_name: agent_parameters[agent_name]}, starting_seed, play_params)

    agent_name = 'ζ-EB'
    starting_seed = 90000

    first_hp = ['beta']+[0.1]+[i for i in range(1, 20, 2)]
    second_hp = ['alpha']+[0.01, 0.05, 0.1, 0.3, 0.5, 1, 1.5, 2, 3]
    fit_parameters_agent(environments, agent_name, nb_iters, first_hp, second_hp,
                         {agent_name: agent_parameters[agent_name]}, starting_seed, play_params)

    agent_name = 'ε-greedy'
    starting_seed = 100000
    first_hp = ['gamma']+[0.95]
    second_hp = ['epsilon']+[0.001]+[0.01]+[0.05] + \
        [round(i*0.1, 1) for i in range(1, 4, 1)] + \
        [round(i*0.1, 1) for i in range(5, 11, 2)]
    fit_parameters_agent(environments, agent_name, nb_iters, first_hp, second_hp,
                         {agent_name: agent_parameters[agent_name]}, starting_seed, play_params)
