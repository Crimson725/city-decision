import os
import time

import gym
import ray
import ray.rllib.agents.dqn as dqn

# noinspection PyUnresolvedReferences
from gym_city.envs.city_env import City_v0
from ray.tune.registry import register_env

# register the custom environment
select_env = "city-v0"
register_env(select_env, lambda config: City_v0())
RAY_DISABLE_MEMORY_MONITOR = 1

# define the algorithm to be used
train_model_name = "dqn"
# path to save the model
chkpt_root = "test/" + train_model_name + "/" + str(int(time.time()))
# path to save the ray results
ray_results = ("{}/ray_results/" + "/" + str(int(time.time()))).format(os.getenv("HOME"))

# training
ray.init(ignore_reinit_error=True)
config = dqn.DEFAULT_CONFIG.copy()
config["log_level"] = "WARN"
agent = dqn.DQNTrainer(config, env=select_env)
# output format
status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}"

# iteration
n_iter = 2

# training begin
for n in range(n_iter):
    result = agent.train()
    chkpt_file = agent.save(chkpt_root)

    print(status.format(
        n + 1,
        result["episode_reward_min"],
        result["episode_reward_mean"],
        result["episode_reward_max"],
        result["episode_len_mean"],
        chkpt_file
    ))

model_path = chkpt_root + "/checkpoint_{}".format(str(n_iter).zfill(6)) + "/checkpoint-{}".format(str(n_iter))


def count_accuracy(ls1, ls2) -> float:
    """
    count the accuracy
    :param ls1: list 1
    :param ls2: list 2
    :return:
    """
    correct_count = 0
    length = len(ls1)
    for i in range(length):
        if ls1[i] == ls2[i]:
            correct_count += 1
    return correct_count / length


# evaluation
def rollout_test(path):
    agent.restore(path)
    env = gym.make(select_env)
    action_list = []
    state = env.reset()
    sum_reward = 0
    n_step = 900
    for step in range(n_step):
        action = agent.compute_single_action(state)
        state, reward, done, info = env.step(action)
        sum_reward += reward
        if (step + 1) % 9 == 0:
            env.render()
            print("final_action", env.get_final_action())
            action_list.append(env.get_final_action())
        if done == 1:
            # report at the end of each episode
            print("cumulative reward", sum_reward)
            state = env.reset()
            sum_reward = 0
    print(count_accuracy(env.label_list, action_list))


rollout_test(model_path)
