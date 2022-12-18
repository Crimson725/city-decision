import os
import time

import gym
import ray
import ray.rllib.agents.dqn as dqn
from ray.tune.registry import register_env
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# noinspection PyUnresolvedReferences
from gym_city.envs.city_world_v3 import City_v3
from gym_city.envs.city_world_test_v3 import City_Test_v3

from expert_reward import get_expert_result

# register custom environment
select_env = "city-test-v3"
test_env = "city-test-v3"
register_env(select_env, lambda config: City_v3())
register_env(test_env, lambda config: City_Test_v3())

RAY_DISABLE_MEMORY_MONITOR = 1

# define the algorithm to be used
train_model_name = "dqn"
# path of the model
chkpt_root = "test/" + train_model_name + "/" + str(int(time.time()))
# ray results
ray_results = ("{}/ray_results/" + "/" + str(int(time.time()))).format(os.getenv("HOME"))

ray.init(ignore_reinit_error=True)
# dqn config
config = dqn.DEFAULT_CONFIG.copy()
config['framework'] = 'torch'
config['noisy'] = True
agent = dqn.DQNTrainer(config, env=select_env)
# output format
status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}"
# interation
n_iter = 2
# print model structure
policy = agent.get_policy()
model = policy.model
# begin training
print("================Begin Training====================")
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

# evaluation
print("================ Evaluations =================")
# model_path='test/dqn/1653376917/checkpoint_000100/checkpoint-100'
model_path = chkpt_root + "/checkpoint_{}".format(str(n_iter).zfill(6)) + "/checkpoint-{}".format(str(n_iter))


# evaluation
def rollout_test():
    # agent.restore(path)
    env = gym.make(test_env)
    state = env.reset()
    sum_reward = 0
    n_step = env.get_data_len() * (20 + 1)
    f1_list = []
    auc_list = []
    sim = []
    cc = []
    kl = []
    for step in range(n_step):
        action = agent.compute_single_action(state)
        state, reward, done, info = env.step(action)
        sim.append(env.get_sim())
        cc.append(env.get_cc())
        kl.append(env.get_kl())
        sum_reward += reward
        if done == 1:
            print("cumulative reward", sum_reward)
            results = env.get_final_action()
            labels = env.label_list
            state = env.reset()
            sum_reward = 0
            print('||------similarity:', sum(sim) / len(sim), '----------||')
            sim.clear()
            print('||------cc:', sum(cc) / len(cc), '----------||')
            cc.clear()
            print('||------kl:', sum(kl) / len(kl), '----------||')
            kl.clear()
            print('||------accuracy:', accuracy_score(labels, results), '----------||')
            print('||------f1_score:', f1_score(labels, results), '----------||')
            f1_list.append(f1_score(labels, results))
            auc_list.append(roc_auc_score(labels, results))
            print('||------auc:', roc_auc_score(labels, results), '----------||')
            # print('||------accuracy:', accuracy_score(pred_expert, results), '----------||')
    print("===========================")
    print("final f1 = ", sum(f1_list) / len(f1_list))
    print("final auc = ", sum(auc_list) / len(auc_list))


rollout_test()
