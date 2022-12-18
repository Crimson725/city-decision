import gym
from gym.utils import seeding

from expert_reward import DesignedPolicy
from utils.rl_utils import *

PATH = 'data/sample.csv'


class City_Test_v3(gym.Env):
    # possible actions
    UNSAFE = 0
    SAFE = 1
    metadata = {
        "render.modes": ["human"]
    }

    def __init__(self):
        # the action space ranges [0, 1] where:
        #  `0` current feature is unsafe
        #  `1` current feature is safe
        self.action_space = gym.spaces.Discrete(2)
        self.dp = DesignedPolicy()

        self.STATE_NUM = self.dp.get_state_num()
        self.observation_space = gym.spaces.Discrete(self.STATE_NUM + 1)

        self.feature_lists = get_images_feature_dict_from_csv(PATH)
        self.data_len = len(self.feature_lists)
        self.filename_list = get_features_from_csv(PATH, 'file_name')
        self.label_list = get_feature_from_csv(PATH, 'is_safe')
        self.reward_dis = []
        # for evaluation
        self.sim = None
        self.cc = None
        self.kl = None

        self.seed()
        self.reset()

    def reset(self) -> int:
        '''
        reset the environment
        :return: initial state
        '''
        self.count = 1
        self.final_action = []
        self.cur_image = 0
        if self.cur_image > self.data_len - 1:
            self.cur_image = self.cur_image % self.data_len

        self.state = self.get_state_from_image()
        self.reward = 0
        self.done = False
        self.info = {}
        return self.state

    def step(self, action)->list:
        '''
        take a step in the environment
        :param action: agent action
        :return: list [state, reward, done, info]
        '''
        if self.done:
            # should never reach this point
            print("EPISODE DONE!!!")
        elif self.count == self.data_len + 1:
            self.done = True
        else:
            assert self.action_space.contains(action)
            # insert simulation logic to handle an action...

            self.reward = self.reward_func(action)
            self.label = self.label_list[self.cur_image]
            self.filename = self.filename_list[self.cur_image][0]
            # print("current image:",self.filename,self.label,self.reward,action)
            # print("reward distribution:",self.reward_dis)
            # get expert reward
            f = self.feature_vector(int(self.label))
            reward_expert = []
            for i in range(len(self.w)):
                reward_expert.append(self.w[i] * f[i])
            self.sim = cal_sim(self.reward_dis, reward_expert)
            self.cc = cal_CC(self.reward_dis, reward_expert)
            self.kl = cal_kl(reward_expert, self.reward_dis)
            self.reward_dis = []
            self.final_action.append(action)
            self.final_reward = self.reward

            # 新的状态
            self.cur_image += 1
            if self.cur_image > self.data_len - 1:
                self.cur_image = self.cur_image % self.data_len
            self.state = self.get_state_from_image()
            # 计数器统一增加
            self.info["dist"] = self.state
            self.count += 1  # 处理图片加一
        try:
            assert self.observation_space.contains(self.state)
        except AssertionError:
            print("INVALID STATE", self.state)
        return [self.state, self.reward, self.done, self.info]

    def get_state_from_image(self)->int:
        '''
        get state from image
        :return: image state
        '''
        self.feature = self.feature_lists[self.cur_image]
        self.w, state, self.state_list = self.dp.image_to_state(self.feature, self.data_len)
        return state

    def feature_vector(self, action)->list:
        '''
        get the feature vector
        :param action: agent action
        :return: list of feature vector
        '''
        feature_vec = []
        for index in range(len(self.state_list)):
            if self.state_list[index] ^ action:
                # 判断正确
                reward = 1
            else:
                reward = -1
            feature_vec.append(reward)
        return feature_vec

    def reward_func(self, action)->int:
        '''
        the designed reward function, can be customized
        :param action: agent action
        :return: reward
        '''
        f = self.feature_vector(action)
        r = 0
        for i in range(len(self.w)):
            self.reward_dis.append(self.w[i] * f[i])
            r += self.w[i] * f[i]

        if action ^ self.label_list[self.cur_image]:
            r = r - 30
        else:
            r = r + 100
        return r

    def get_final_action(self)->int:
        '''
        get the final action
        :return: int
        '''
        return self.final_action

    def get_data_len(self)->int:
        '''
        get the length of the dataset
        :return: int
        '''
        return self.data_len

    def get_sim(self)->float:
        '''
        get the similarity for evaluation
        :return: float
        '''
        return self.sim

    def get_cc(self)->float:
        '''
        get the cc value for evaluation
        :return: float
        '''
        return self.cc

    def get_kl(self)-> float:
        '''
        get the kl divergence for evaluation
        :return: float
        '''
        return self.kl

    def render(self, mode="human"):
        s = "position: {:2d}  reward: {:2f}  label: {}  action: {}  filename:{} "
        print(s.format(self.state, self.reward, self.label, self.final_action[-1], self.filename))
        # print("{} is final action in env, label is {} and the reward is {}".format(self.final_action, self.feature[8], self.final_reward))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        pass
