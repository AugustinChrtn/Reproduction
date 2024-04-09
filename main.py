"""
Generate all the simulations. Use multiprocessing and run faster on machines with multiple cores.

"""

from main_functions import evaluate_agents

if __name__ == "__main__":
    print("""This program launches all the experiments presented in the article.
          
          There are 15 experiments with different parameters found using
          the parameter_fitting.py program.
          
          For each experiment, we print the computational time the experiment
          took. 

          This code uses multiprocessing and runs faster on machines with many
          cores. 

          An estimation of the total computational time is 60 times the time
          of the first experiment (e.g 20 min if the first experiment takes 20
          seconds.)

          """)
    # agents=['R-max','ζ-R-max','BEB','ζ-EB','ε-greedy']
    agents = ['R-max', 'ζ-R-max', 'BEB', 'ζ-EB',
              'ε-greedy']  # All the agents to plot

    # Parameters to use for the play function
    play_parameters = {'trials': 100, 'max_step': 30, 'screen': False,
                       'photos': [], 'accuracy_VI': 0.001, 'step_between_VI': 50}

    # Environment of Lopes et al. (2012)

    # Parameters found using parameter fitting for Lopes et al.'s metrics
    agent_param_opt = {
        'ε-greedy': {'gamma': 0.95, 'epsilon': 0.3},
        'R-max': {'gamma': 0.95, 'm': 8, 'Rmax': 1, 'm_u': 12, 'condition': 'informative'},
        'BEB': {'gamma': 0.95, 'beta': 7, 'coeff_prior': 2, 'condition': 'informative'},
        'ζ-EB': {'gamma': 0.95, 'beta': 3, 'step_update': 10, 'alpha': 0.1, 'prior_LP': 0.01},
        'ζ-R-max': {'gamma': 0.95, 'Rmax': 1, 'm': 2, 'step_update': 10, 'alpha': 0.3, 'prior_LP': 0.01}}

    # Stationary environment
    environments = ['Lopes']
    nb_iters = 20
    starting_seed = 100

    evaluate_agents(environments, agents, nb_iters, play_parameters,
                    agent_param_opt, starting_seed,
                    title_fig='Figure_2_(or_11_top_left)_good_prior_optimal_policy_parameters')

    # Wrong prior
    starting_seed = 200
    agent_param_opt['R-max']['condition'] = 'wrong_prior'
    agent_param_opt['BEB']['condition'] = 'wrong_prior'

    evaluate_agents(environments, agents, nb_iters, play_parameters,
                    agent_param_opt, starting_seed,
                    title_fig='Figure_3_(or_11_second_left)_wrong_prior_optimal_policy_parameters')

    agent_param_opt['R-max']['condition'] = 'informative'
    agent_param_opt['BEB']['condition'] = 'informative'

    # Non-stationarity from the article
    starting_seed = 300
    nb_iters = 5
    environments = ["Lopes_non_stat_article_{0}".format(
        ind_non_stat)for ind_non_stat in range(1, 21)]

    evaluate_agents(environments, agents, nb_iters, play_parameters,
                    agent_param_opt, starting_seed,
                    title_fig='Figure_5_(or_11_fourth_left)_small_non_stationarity_optimal_policy_parameters')

    # Stronger non-stationarity

    starting_seed = 400
    environments = ["Lopes_strong_non_stat_{0}".format(
        ind_non_stat) for ind_non_stat in range(1, 21)]

    evaluate_agents(environments, agents, nb_iters, play_parameters, agent_param_opt, starting_seed,
                    title_fig='Figure_6_(or_11_last_left)_strong_non_stationarity_optimal_policy_parameters')

    starting_seed = 400
    play_parameters['trials'] = 200
    evaluate_agents(environments, agents, nb_iters, play_parameters, agent_param_opt, starting_seed,
                    title_fig='Figure_6_strong_non_stationarity_long_horizon_optimal_policy_parameters')
    play_parameters['trials'] = 100

    # Uninformative prior

    environments = ['Lopes']
    nb_iters = 20
    starting_seed = 250

    agent_param_opt['R-max'] = {'gamma': 0.95, 'm': 10,
                                'Rmax': 1, 'm_u': 10, 'condition': 'uniform'}
    agent_param_opt['BEB'] = {'gamma': 0.95, 'beta': 7,
                              'coeff_prior': 0.001, 'condition': 'uniform'}

    evaluate_agents(environments, agents, nb_iters, play_parameters, agent_param_opt, starting_seed,
                    title_fig='Figure_4_(or_11_middle_left)_uninformative_prior_optimal_policy_parameters')

    # With the agent parameters to maximize the real agent policy

    agent_param_real = {
        'ε-greedy': {'gamma': 0.95, 'epsilon': 0.01},
        'R-max': {'gamma': 0.95, 'm': 8, 'Rmax': 1, 'm_u': 12, 'condition': 'informative'},
        'BEB': {'gamma': 0.95, 'beta': 3, 'coeff_prior': 2, 'condition': 'informative'},
        'ζ-EB': {'gamma': 0.95, 'beta': 1, 'step_update': 10, 'alpha': 0.1, 'prior_LP': 0.01},
        'ζ-R-max': {'gamma': 0.95, 'Rmax': 1, 'm': 2, 'step_update': 10, 'alpha': 0.1, 'prior_LP': 0.01}}

    # stationary
    environments = ["Lopes"]
    nb_iters = 20

    starting_seed = 500
    evaluate_agents(environments, agents, nb_iters, play_parameters,
                    agent_param_real, starting_seed,
                    title_fig='Figure_11_top_right_real_policy_parameters')

    # wrong prior
    starting_seed = 600
    agent_param_real['R-max']['condition'] = 'wrong_prior'
    agent_param_real['BEB']['condition'] = 'wrong_prior'

    evaluate_agents(environments, agents, nb_iters, play_parameters,
                    agent_param_real, starting_seed,
                    title_fig='Figure_11_second_right_wrong_prior_real_policy_parameters')

    # Reproduction of the third figure of Lopes et al. (2012)
    starting_seed = 300
    nb_iters = 5
    agent_param_real['R-max']['condition'] = 'informative'
    agent_param_real['BEB']['condition'] = 'informative'

    environments = ["Lopes_non_stat_article_{0}".format(
        ind_non_stat)for ind_non_stat in range(1, 21)]

    evaluate_agents(environments, agents, nb_iters, play_parameters,
                    agent_param_real, starting_seed,
                    title_fig='Figure_11_fourth_right_small_non_stationarity_real_policy_parameters')

    # Stronger non-stationarity to find the same third figure as Lopes et al. (2012)
    starting_seed = 400
    environments = ["Lopes_strong_non_stat_{0}".format(
        ind_non_stat) for ind_non_stat in range(1, 21)]

    evaluate_agents(environments, agents, nb_iters, play_parameters,
                    agent_param_real, starting_seed,
                    title_fig='Figure_11_bottom_right_strong_non_stationarity_real_policy_parameters')

    # Without informative prior
    starting_seed = 250
    environments = ["Lopes"]
    nb_iters = 20

    agent_param_real['R-max'] = {'gamma': 0.95, 'm': 10,
                                 'Rmax': 1, 'm_u': 10, 'condition': 'uniform'}
    agent_param_real['BEB'] = {'gamma': 0.95, 'beta': 3,
                               'coeff_prior': 0.001, 'condition': 'uniform'}

    evaluate_agents(environments, agents, nb_iters, play_parameters,
                    agent_param_real, starting_seed,
                    title_fig='Figure_11_middle_right_uniform_prior_real_policy_parameters')

    # Replication with negative reward of -1

    agent_param_opt = {
        'ε-greedy': {'gamma': 0.95, 'epsilon': 0.3},
        'R-max': {'gamma': 0.95, 'm': 8, 'Rmax': 1, 'm_u': 12, 'condition': 'informative'},
        'BEB': {'gamma': 0.95, 'beta': 7, 'coeff_prior': 2, 'condition': 'informative'},
        'ζ-EB': {'gamma': 0.95, 'beta': 3, 'step_update': 10, 'alpha': 0.1, 'prior_LP': 0.01},
        'ζ-R-max': {'gamma': 0.95, 'Rmax': 1, 'm': 2, 'step_update': 10, 'alpha': 0.3, 'prior_LP': 0.01}}

    # Stationary environment
    starting_seed = 1000

    environments = ['Stationary_-1_'+str(number_world)
                    for number_world in range(1, 11)]
    nb_iters = 10
    evaluate_agents(environments, agents, nb_iters, play_parameters,
                    agent_param_opt, starting_seed,
                    title_fig='Figure_13_top_left_optimal_policy_parameters_punish_-1')

    # Wrong prior
    starting_seed = 1200
    agent_param_opt['R-max']['condition'] = 'wrong_prior'
    agent_param_opt['BEB']['condition'] = 'wrong_prior'

    evaluate_agents(environments, agents, nb_iters, play_parameters,
                    agent_param_opt, starting_seed,
                    title_fig='Figure_13_top_right_wrong_prior_optimal_policy_parameters_punish_-1')

    agent_param_opt['R-max']['condition'] = 'informative'
    agent_param_opt['BEB']['condition'] = 'informative'

    # Small non-stationarity
    starting_seed = 1500
    nb_iters = 1
    environments = ["Non_stat_article_-1_{0}".format(nb_env)+'_{0}'.format(ind_non_stat)
                    for nb_env in range(1, 11) for ind_non_stat in range(1, 11)]
    evaluate_agents(environments, agents, nb_iters, play_parameters,
                    agent_param_opt, starting_seed,
                    title_fig='Figure_13_bottom_left_small_non_stationarity_optimal_policy_parameters_punish_-1')

    # Strong non-stationarity
    starting_seed = 2000
    environments = ["Strong_non_stat_-1_{0}".format(nb_env)+'_{0}'.format(ind_non_stat)
                    for nb_env in range(1, 11) for ind_non_stat in range(1, 11)]
    evaluate_agents(environments, agents, nb_iters,
                    play_parameters, agent_param_opt, starting_seed,
                    title_fig ='Figure_13_bottom_right_strong_non_stationarity_optimal_policy_parameters_punish_-1')
