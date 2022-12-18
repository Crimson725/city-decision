import gym
import ray
# use D3QN (default D3QN)
import ray.rllib.agents.dqn as dqn
# noinspection PyUnresolvedReferences
from gym_city.envs.city_world_v3 import City_v3
from ray.tune.registry import register_env
from sklearn.metrics import accuracy_score

# needs to define the training set and test set
select_env = "city-v3"
test_env = "city-v3"

register_env(select_env, lambda config: City_v3())
register_env(test_env, lambda config: City_v3())
# prevent crash
RAY_DISABLE_MEMORY_MONITOR = 1
ray.init(ignore_reinit_error=True)
# D3QN config (can be finetuned)
config = dqn.DEFAULT_CONFIG.copy()
config['framework'] = 'torch'
# set noisy for efficient exploring
config['noisy'] = True
agent = dqn.DQNTrainer(config, env=select_env)


def rollout_test(path):
    """
    rollout the model and get the evaluation results
    :param path: the path of the model
    :return:
    """
    agent.restore(path)
    env = gym.make(select_env)
    state = env.reset()
    sum_reward = 0
    n_step = env.get_data_len() * (5 + 1)
    for step in range(n_step):
        action = agent.compute_single_action(state)
        state, reward, done, info = env.step(action)
        sum_reward += reward
        if done == 1:
            # report at the end of each episode
            print("cumulative reward", sum_reward)
            results = env.get_final_action()
            labels = env.label_list
            state = env.reset()
            sum_reward = 0
            print('||------accuracy:', accuracy_score(labels, results), '----------||')


# model path
model_path = 'trained_model/trained_model'
if __name__ == '__main__':
    rollout_test(model_path)
