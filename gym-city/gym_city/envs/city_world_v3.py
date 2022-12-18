import gym
from gym.utils import seeding

from expert_reward import DesignedPolicy
from utils.rl_utils import *

# PATH='data/sample.csv'
PATH = 'data/train_random1.csv'


class City_v3(gym.Env):
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

        # change to guarantee the sequence of pseudorandom numbers
        # (e.g., for debugging)
        self.seed()
        self.reset()

    def reset(self) -> int:
        '''
        rest the environment
        :return: initial state
        '''
        self.count = 1
        self.final_action = []  # save the final action
        self.cur_image = 0  # first state
        if self.cur_image > self.data_len - 1:
            self.cur_image = self.cur_image % self.data_len

        self.state = self.get_state_from_image()
        self.reward = 0
        self.done = False
        self.info = {}
        return self.state

    def step(self, action) -> list:
        '''
        take a step in the environment
        :param action: agent action
        :return: list of [state, reward, done, info]
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
            self.final_action.append(action)
            self.final_reward = self.reward

            # new state
            self.cur_image += 1
            if self.cur_image > self.data_len - 1:
                self.cur_image = self.cur_image % self.data_len
            self.state = self.get_state_from_image()
            # counter
            self.info["dist"] = self.state
            self.count += 1
        try:
            assert self.observation_space.contains(self.state)
        except AssertionError:
            print("INVALID STATE", self.state)
        return [self.state, self.reward, self.done, self.info]

    def get_state_from_image(self) -> int:
        '''
        get state from iamge
        :return: the image state
        '''
        self.feature = self.feature_lists[self.cur_image]
        self.w, state, self.state_list = self.dp.image_to_state(self.feature, self.data_len)
        return state

    def feature_vector(self, action) -> list:
        '''
        get the feature vector
        :param action: agent action
        :return: list of feature vec
        '''
        feature_vec = []
        for index in range(len(self.state_list)):
            if self.state_list[index] ^ action:
                reward = 1
            else:
                reward = -1
            feature_vec.append(reward)
        return feature_vec

    def reward_func(self, action) -> int:
        '''
        the designed reward function, can be customized
        :param action: agent action
        :return: reward
        '''
        f = self.feature_vector(action)
        r = 0
        for i in range(len(self.w)):
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

    def get_expert_result(self)->float:
        '''
        get optiomal expert result
        :return: float
        '''
        return self.dp.get_optimal_result()

    def render(self, mode="human"):
        s = "position: {:2d}  reward: {:2f}  label: {}  action: {}  filename:{} "
        print(s.format(self.state, self.reward, self.label, self.final_action[-1], self.filename))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        pass
