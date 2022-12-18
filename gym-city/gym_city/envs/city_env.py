import gym
from gym.utils import seeding
from utils.rl_utils import *
from params import *

PATH = "data/sample.csv"


class City_v0(gym.Env):
    # possible actions
    UNSAFE = 0
    SAFE = 1

    START_FEA = 1
    LAST_FEA = 9  # last step

    # land on the GOAL position within MAX_STEPS steps
    MAX_STEPS = 8

    metadata = {
        "render.modes": ["human"]
    }

    def __init__(self):
        # the action space ranges [0, 1] where:
        #  `0` current feature is unsafe
        #  `1` current feature is safe
        self.action_space = gym.spaces.Discrete(2)

        self.observation_space = gym.spaces.Discrete(self.LAST_FEA + 2)
        self.feature_lists = get_features_from_csv(PATH, 'building', 'complexity', 'isline', 'wall', 'car_num',
                                                   'greenery', 'sky', 'people_num', 'is_safe')
        self.filename_list = get_features_from_csv(PATH, 'file_name')
        self.label_list = get_feature_from_csv(PATH, 'is_safe')
        self.cur_image = -2  # current episode
        self.goal = self.LAST_FEA

        self.init_features = [1]
        self.building_mode = 0
        self.final_action = 10
        self.arg_th = 0.6
        self.seed()
        self.reset()

    def reset(self)->int:
        '''
        rest the environment
        :return: initial state
        '''
        self.position = self.np_random.choice(self.init_features)  # current feature position
        self.count = 0
        self.cur_image += 1  #

        if self.cur_image > 99:
            self.cur_image = self.cur_image % 100
        self.feature = self.feature_lists[self.cur_image]
        self.filename = self.filename_list[self.cur_image][0]
        self.label = self.label_list[self.cur_image]

        self.choice = []

        self.state = self.position
        self.reward = 0
        self.done = False
        self.info = {}
        return self.state

    def step(self, action)->int:
        '''
        environment step
        :param action:
        :return:
        '''
        if self.done:
            # should never reach this point
            print("EPISODE DONE!!!")
        elif self.count == self.LAST_FEA:
            self.done = True
        else:
            assert self.action_space.contains(action)
            self.choice.append(action)
            self.count += 1
            # insert simulation logic to handle an action...
            if self.count == 1:
                self.reward = 0
                self.position += 1
                if self.feature[0] >= 0.15:
                    self.building_mode = 1
                else:
                    self.building_mode = 0
            else:
                if self.position == self.goal:
                    self.done = True
                    action = self.get_final_decision()
                    self.final_action = action
                    if action == int(self.feature[8]):
                        self.reward = 5
                    else:
                        self.reward = -5
                    self.position += 1

                elif self.position == 2:
                    cur_feature_value = self.feature[self.position - 1]
                    if (cur_feature_value > 0 and action == self.UNSAFE) or (
                            int(cur_feature_value) == 0 and action == self.SAFE):
                        self.reward = 7
                    else:
                        self.reward = -7
                    self.position += 1
                elif self.position == 3:
                    cur_feature_value = self.feature[self.position - 1]
                    if (cur_feature_value > 0 and action == self.UNSAFE) or (
                            int(cur_feature_value) == 0 and action == self.SAFE):
                        self.reward = 3
                    else:
                        self.reward = -3
                    self.position += 1
                elif self.position == 4:
                    # car_num
                    cur_feature_value = self.feature[self.position - 1]
                    if cur_feature_value >= car_num_th and action == self.UNSAFE:
                        self.reward = 3 / (cur_feature_value - 2)
                    elif cur_feature_value >= car_num_th and action == self.SAFE:
                        self.reward = -0.3 * (cur_feature_value - 2)
                    elif cur_feature_value < car_num_th and action == self.UNSAFE:
                        self.reward = -3 * (3 - cur_feature_value)
                    else:
                        self.reward = 3 / (3 - cur_feature_value)
                    self.position += 1
                elif self.position == 5:
                    # sky
                    cur_feature_value = self.feature[self.position - 1]
                    if self.building_mode:
                        if cur_feature_value <= sky_th_1:
                            if action == self.UNSAFE:
                                self.reward = 10 * 0.01 / (0.2 - cur_feature_value)
                            else:
                                self.reward = -10 * (0.2 - cur_feature_value)
                        else:
                            if action == self.UNSAFE:
                                self.reward = -10 * (cur_feature_value - 0.2)
                            else:
                                self.reward = 0.01 / (cur_feature_value - 0.2)
                    else:
                        if cur_feature_value <= sky_th_0:
                            if action == self.UNSAFE:
                                self.reward = 3 * 0.35 / ((0.35 - cur_feature_value) * 1000)
                            else:
                                self.reward = -3 * (0.35 - cur_feature_value)
                        else:
                            if action == self.UNSAFE:
                                self.reward = -3 * (cur_feature_value - 0.35)
                            else:
                                self.reward = 3 * 0.35 / ((cur_feature_value - 0.35) * 1000)
                    self.position += 1
                elif self.position == 6:
                    # greenery
                    cur_feature_value = self.feature[self.position - 1]
                    if self.building_mode:
                        if cur_feature_value <= greenery_th_1:
                            if action == self.UNSAFE:
                                self.reward = 0.001 / (0.15 - cur_feature_value)
                            else:
                                self.reward = -100 * (0.15 - cur_feature_value)
                        else:
                            if action == self.UNSAFE:
                                self.reward = -100 * (cur_feature_value - 0.15)
                            else:
                                self.reward = 0.001 / (cur_feature_value - 0.15)
                    else:
                        if cur_feature_value >= greenery_th_0:
                            if action == self.UNSAFE:
                                self.reward = 0.001 / (cur_feature_value - 0.2)
                            else:
                                self.reward = -100 * (cur_feature_value - 0.2)
                        else:
                            if action == self.UNSAFE:
                                self.reward = -100 * (0.2 - cur_feature_value)
                            else:
                                self.reward = 0.001 / (0.2 - cur_feature_value)
                    self.position += 1
                elif self.position == 7:
                    # people_num
                    cur_feature_value = self.feature[self.position - 1]
                    if cur_feature_value <= people_num_th:
                        if action == self.UNSAFE:
                            self.reward = 1 / (3 - cur_feature_value)
                        else:
                            self.reward = -3 * (3 - cur_feature_value)
                    else:
                        if action == self.UNSAFE:
                            self.reward = -0.3 * (cur_feature_value - 2)
                        else:
                            self.reward = 1 / (cur_feature_value - 2)
                    self.position += 1
                elif self.position == 8:
                    # complexity
                    cur_feature_value = self.feature[self.position - 1]
                    if self.building_mode:
                        # self.reward = self.expert_to_r(complexity_th_1, 1, action)
                        if cur_feature_value >= complexity_th_1:
                            if action == self.UNSAFE:
                                self.reward = 2 * 0.0169 / (cur_feature_value - complexity_th_1)
                            else:
                                self.reward = -2 * (cur_feature_value - complexity_th_1)
                        else:
                            if action == self.UNSAFE:
                                self.reward = -2 * (complexity_th_1 - cur_feature_value)
                            else:
                                self.reward = 2 * 0.0169 / (complexity_th_1 - cur_feature_value)
                    else:
                        if cur_feature_value >= complexity_th_0:
                            if action == self.UNSAFE:
                                self.reward = 2 * 0.00116 / (cur_feature_value - complexity_th_0)
                            else:
                                self.reward = -2 * (cur_feature_value - complexity_th_0)
                        else:
                            if action == self.UNSAFE:
                                self.reward = -2 * (complexity_th_0 - cur_feature_value)
                            else:
                                self.reward = 2 * 0.0116 / (complexity_th_0 - cur_feature_value)
                        # self.reward = self.expert_to_r(complexity_th_0, 1, action)
                    self.position += 1

            self.state = self.position
            self.info["dist"] = self.goal - self.position

        try:
            assert self.observation_space.contains(self.state)
        except AssertionError:
            print("INVALID STATE", self.state)
        return [self.state, self.reward, self.done, self.info]

    def expert_to_r(self, threshold, flag, action) -> list:
        """
        count : current feature (1-based)
        :param threshold:
        :param flag: 1 upper threshold; 0 lower threshold
        :return: reward[]
        """
        penalty = self.get_penalty()  # 获取惩罚值
        cur_feature_value = self.feature[self.count - 1]
        reward = 0
        if flag:
            if cur_feature_value >= threshold and action == self.UNSAFE:
                reward = -penalty * (threshold / (cur_feature_value - threshold))
            elif cur_feature_value >= threshold and action == self.SAFE:
                reward = penalty * (cur_feature_value - threshold)
            elif cur_feature_value < threshold and action == self.UNSAFE:
                reward = penalty * (cur_feature_value - threshold)
            elif cur_feature_value < threshold and action == self.SAFE:
                reward = -penalty * (threshold / (threshold - cur_feature_value))
            else:
                pass
        else:
            if cur_feature_value <= threshold and action == self.UNSAFE:
                reward = -penalty * (threshold / (threshold - cur_feature_value))  # 正向reward-----该处的函数需要设计？？？？-----
            elif cur_feature_value <= threshold and action == self.SAFE:
                reward = penalty * (threshold - cur_feature_value)
            elif cur_feature_value > threshold and action == self.UNSAFE:
                reward = penalty * (cur_feature_value - threshold)
            elif cur_feature_value > threshold and action == self.SAFE:
                reward = -penalty * (threshold / (cur_feature_value - threshold))
            else:
                pass
        return reward

    def get_penalty(self) -> int:
        '''
        get penalty in different building_mode
        :return: penalty
        '''
        if self.building_mode:
            if self.count == 2:
                return -7
            elif self.count == 3:
                return -3
            elif self.count == 4:
                return -1
            elif self.count == 5:
                return -1
            elif self.count == 6:
                return -1
            elif self.count == 8:
                return -2
            else:
                return 0
        else:
            if self.count == 8:
                return -2
            elif self.count == 6:
                return -1
            elif self.count == 5:
                return -3
            elif self.count == 7:
                return -1
            else:
                return 0

    def get_final_decision(self) -> int:
        '''
        get final decision
        :return: 1: safe; 0: unsafe
        '''
        k_1 = [0, 0.467, 0.2, 0.067, 0.067, 0.067, 0, 0.133]  # weights of the features
        k_0 = [0, 0, 0, 0, 0.429, 0.143, 0.143, 0.286]
        th = 0  # 阈值
        if self.building_mode:
            for i in range(self.LAST_FEA - 1):
                th = th + self.choice[i] * k_1[i]
        else:
            for i in range(self.LAST_FEA - 1):
                th = th + self.choice[i] * k_0[i]
        if th > self.arg_th:
            return self.SAFE
        else:
            return self.UNSAFE

    def ren1der(self, mode="human"):
        '''
        render the environment (only output)
        :param mode:
        :return:
        '''
        s = "position: {:2d}  reward: {:2f}  label: {} filename:{} "
        print(s.format(self.state, self.reward, self.label, self.filename))
        if self.state == 10:
            print("{} is action in env, label is {}".format(self.final_action, self.feature[8]))

    def get_final_action(self) -> int:
        '''
        get the final action (used for evaluation)
        :return: final action
        '''
        if self.state == 10:
            return self.final_action

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        pass
