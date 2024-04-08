"""Generate all the environments used for the simulations."""

from lopesworld import Lopes_environment
from policy_functions import value_iteration
import numpy as np

# UP, DOWN, LEFT, RIGHT, STAY = 0, 1, 2, 3, 4


def transition_Lopes(alpha_max=1, alpha_min=0.1):
    """
    Generate transition of a Lopes and colleagues' (2012) environment.

    Parameters
    ----------
    alpha_max : float (>0)
        alpha value for the Dirichlet distributions of uncertain states.
    alpha_min : float (>0)
        alpha value for the Dirichlet distributions of states that are less uncertain.

    Returns
    -------
    transitions : numpy.ndarray of shape (size_state,size_action,size_state)
        transition array for a stationary Lopes and colleagues environment.

    """
    assert alpha_min > 0, "alpha values need to be stricly positive"
    assert alpha_max > 0, "alpha values need to be stricly positive"

    transitions = np.zeros((25, 5, 25))
    uncertain_states = [1, 3, 11, 13]

    for state in range(25):
        if state in uncertain_states:
            alpha = alpha_max
        else:
            alpha = alpha_min

        reach_state_bool = np.array([state > 4, state < 20, state % 5 != 0, state % 5 != 4, True])
        result_of_the_action = np.array([-5, +5, -1, 1, 0])
        reachable_states = np.unique(state+reach_state_bool*result_of_the_action)
        for action in range(5):
            values = np.random.dirichlet([alpha]*len(reachable_states))
            deterministic_state = state+reach_state_bool[action]*result_of_the_action[action]
            for index_arrival_state, arrival_state in enumerate(reachable_states):
                transitions[state, action, arrival_state] = values[index_arrival_state]
            if transitions[state, action, deterministic_state] != np.max(values):
                state_max = np.argmax(transitions[state, action])
                tSA = transitions[state, action]
                tSA[deterministic_state], tSA[state_max] = tSA[state_max], tSA[deterministic_state]
    return transitions


STATES_OPTIMAL_PATH = [0, 5, 15, 16, 17, 18, 19, 14]


def non_stat_Lopes_article(transitions, index_state_to_change):
    """Change the transitions of one state on the optimal path."""
    new_transitions = np.copy(transitions)
    changed_state = STATES_OPTIMAL_PATH[index_state_to_change]

    derangement_list = np.arange(5)
    while np.any(derangement_list == np.arange(5)):
        derangement_list = np.random.permutation(derangement_list)

    for action in range(5):
        changed_transitions = transitions[changed_state][action]
        new_transitions[changed_state][derangement_list[action]] = changed_transitions
    return new_transitions


def non_stat_Lopes_all_states(transitions):
    """Change the transitions of all the states on the optimal path."""
    for index in range(len(STATES_OPTIMAL_PATH)):
        transitions = non_stat_Lopes_article(transitions, index_state_to_change=index)
    return transitions

# Reward generation


def reward_Lopes(bonus=1, malus=-0.1):
    """Generate an array of rewards for Lopes and colleagues' environment."""
    array_rewards = np.zeros(25)
    lopes_rewards = {6: malus, 7: malus, 8: malus, 12: malus, 14: bonus}
    for state, reward in lopes_rewards.items():
        array_rewards[state] = reward
    return array_rewards


def save_rewards(bonus=1, malus=-0.1):
    """Save the numpy array of the rewards."""
    array_rewards = reward_Lopes(bonus, malus)
    np.save('Environments/Rewards_'+str(malus)+'.npy', array_rewards)


# Validity of the worlds generated


def valid_policy(policy):
    """Check if a policy is the same as the optimal one from Lopes et al.'s article."""
    return (policy[0] == 1 and policy[5] == 1 and policy[10] == 1 and policy[15] == 3 and
            policy[16] == 3 and policy[17] == 3 and policy[18] == 3 and policy[19] == 0)


def generate_Lopes_environment(alpha_max=1, alpha_min=0.1, malus=-0.1, bonus=1):
    """Generate an environment."""
    transitions = transition_Lopes(alpha_max, alpha_min)
    rewards = reward_Lopes(malus=malus, bonus=bonus)
    environment = Lopes_environment(transitions, rewards)
    return environment, transitions


def valid_environment(environment, gamma=0.95, threshold_VI=1e-3):
    """Check if the optimal path of an environment is valid."""
    _, policy = value_iteration(environment, gamma, threshold_VI)
    return valid_policy(policy)

# Statistics


def proportion_of_valid_worlds(iterations=5000, alpha_max=1, alpha_min=0.1, malus=-0.1, bonus=1):
    """Print the percentage of valid environments out of :iterations: tries."""
    valid_count = 0
    for i in range(iterations):
        environment, transitions = generate_Lopes_environment(alpha_max, alpha_min, malus, bonus)
        validity = valid_environment(environment)
        valid_count += validity
    percentage_valid = round(100*valid_count/iterations, 1)
    print('')
    print('Percentage of valid environments out of ' + str(iterations) +
          ' worlds, with a malus of ' + str(malus)+': '
          + str(percentage_valid)+'%')


def generate_valid_env(nb_env=1, alpha_max=1, alpha_min=0.1, malus=-0.1, bonus=1):
    """Generate :nb_env: valid environments."""
    for index_world in range(1, nb_env + 1):
        counter, validity = 0, False
        while not validity and counter < 1000:
            env, transitions = generate_Lopes_environment(alpha_max, alpha_min, malus, bonus)
            validity = valid_environment(env)
            counter += 1
        if counter == 1000:
            raise RuntimeError('No valid world found in '+str(counter)+' iterations.')
        np.save('Environments/Transitions_' + str(malus) + '_'
                + str(index_world) + '.npy', transitions)


def generate_non_stationarity_article(world_number=1, nb_env=1, malus=-0.1):
    """Generate non-stationary environments based on previously generated valid environments."""
    for index_world in range(1, world_number+1):
        transitions = np.load('Environments/Transitions_'+str(malus) +
                              '_'+str(index_world)+'.npy', allow_pickle=True)
        for non_stat_number in range(1, nb_env+1):
            transitions_non_stationary_article = non_stat_Lopes_article(
                transitions, np.random.randint(len(STATES_OPTIMAL_PATH)))
            np.save('Environments/Transitions_non_stat_article_'+str(malus)+'_'+str(index_world) +
                    '_'+str(non_stat_number)+'.npy', transitions_non_stationary_article)


def generate_strong_non_stationarity_(world_number=1, nb_env=1, malus=-0.1):
    """Generate strong non_stationary environments."""
    for index_world in range(1, world_number + 1):
        transitions = np.load('Environments/Transitions_' + str(malus) +
                              '_'+str(index_world) + '.npy', allow_pickle=True)
        for non_stat_number in range(1, nb_env+1):
            transitions_strong_non_stationarity = non_stat_Lopes_all_states(transitions)
            np.save('Environments/Transitions_strong_non_stat_'+str(malus)+'_'+str(index_world) +
                    '_' + str(non_stat_number) + '.npy', transitions_strong_non_stationarity)

if __name__=="__main__":
# Generate the (deterministic) reward matrices
    print("This program generates all the environments.")   
    save_rewards(1, -0.1)
    save_rewards(1, -1)

    # Generate an environment similar to the one of Lopes and colleagues
    np.random.seed(1)
    generate_valid_env(nb_env=1, alpha_max=1, alpha_min=0.1, malus=-0.1, bonus=1)

    # Generate 20 environments with one change in the optimal path
    np.random.seed(2)
    generate_non_stationarity_article(world_number=1, nb_env=20, malus=-0.1)

    # Generate 20 environments with changes for all the states in the optimal path
    np.random.seed(3)
    generate_strong_non_stationarity_(world_number=1, nb_env=20, malus=-0.1)

    # Generate 10 stationary environments with Bureau & Sebag (2013) parameters
    np.random.seed(4)
    generate_valid_env(nb_env=10, alpha_max=1, alpha_min=0.1, malus=-1, bonus=1)

    # Generate 10 envs for each stationary env, with one change in the optimal path
    np.random.seed(5)
    generate_non_stationarity_article(world_number=10, nb_env=10, malus=-1)

    # Generate 10 envs for each stationary env, with changes for all the states in the optimal path
    np.random.seed(6)
    generate_strong_non_stationarity_(world_number=10, nb_env=10, malus=-1)

    print("Success: All the environments have been generated.")
    print("")
    print("Computing stats depending on the value of the punishment." )
    # Check the percentage of valid environments for each condition
    np.random.seed(10)
    proportion_of_valid_worlds(iterations=5000, alpha_max=1, alpha_min=0.1, malus=-0.1, bonus=1)
    proportion_of_valid_worlds(iterations=5000, alpha_max=1, alpha_min=0.1, malus=-1, bonus=1)
    print("Success: End of computation." )
