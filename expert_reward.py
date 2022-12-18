from utils.rl_utils import get_images_feature_dict_from_csv
import math
import numpy as np
from utils.common_utils import list_to_int


# --------------------------------------------------------------------
# ---------------------------Policy API-------------------------------
# --------------------------------------------------------------------
# You can modify these code to design your own evaluation policy.
def get_expert_policy() -> (list, list):
    """
    get the expert policy
    :return: list: functionArea, list: penalty
    """
    functionArea = ['business', 'culture', 'residence', 'industry', 'outdoors', 'else']
    # [sign:>-1;<-0  , threshold, penalty]
    area_0 = {
        'greenery': [1, 0.22209370183333332, -1],
        'wall': [1, 0, -3],
        'traffic_light': [0, 1, -4],
        'traffic_sign': [0, 1, -4],
        'isline': [1, 0, -5],
        'sky': [1, 0.2212055845, -2],
        'building': [1, 0.1545, -1]
    }

    area_1 = {
        'greenery': [1, 0.28, -2],
        'wall': [1, 0, -3],
        'isline': [1, 0, -5],
        'sky': [0, 0.067, -1],
        'person': [0, 3, -1]
    }

    area_2 = {
        'isline': [1, 0, -5],
        'wall': [1, 0, -2],
        'building': [1, 0.43, -1],
        'sky': [0, 0.1, -1],
        'visual_entropy': [1, 0.46, -2],
        'vegetation': [1, 0.1431, -1],
        'car_num': [1, 4, -3]
    }

    area_3 = {
        'isline': [1, 0, -2],
        'wall': [1, 0, -2],
        'car_num': [0, 5, -1],
        'sidewalk': [0, 0, -3],
        'greenery': [0, 0.091, -2]
    }

    area_4 = {
        'isline': [1, 0, -5],
        'wall': [1, 0, -2],
        'sky': [0, 0.115, -2],
        'visual_entropy': [1, 0.4, -2]
    }

    area_5 = {
        'visual_entropy': [1, 0.5199, -2],
        'isline': [1, 0, -5],
        'building': [1, 0.1777, -1],
        'vegetation': [1, 0.1686, -1],
        'wall': [1, 0, -3],
        'sky': [0, 0.25, -1],
        'car_num': [1, 5, -3]
    }
    # penalty of the function area
    areaPenalty = [area_0, area_1, area_2, area_3, area_4, area_5]
    return functionArea, areaPenalty


def get_expert_result(path) -> list:
    """
    get the expert result
    :param path: path of the csv file
    :return: expert results list
    """
    feature_lists = get_images_feature_dict_from_csv(path)
    dp = DesignedPolicy()
    for item in feature_lists:
        _, _, _ = dp.image_to_state(item, len(feature_lists))
    penalty_scores = dp.get_optimal_result()
    th = - sum(penalty_scores) / len(penalty_scores)  # 平均值的相反数，化为正数
    alpha_1 = -1.000001
    alpha_2 = -0.999999
    results = []
    for score in penalty_scores:
        if score > alpha_2 * th:
            results.append(1)
        elif score < alpha_1 * th:
            results.append(0)
        else:
            s = np.random.randint(0, 2)
            results.append(s)
    print("results for expert:", results)
    return results


class DesignedPolicy:
    def __init__(self):
        self.FunctionArea, self.AreaPenalty = get_expert_policy()
        self.function_num = len(self.FunctionArea)
        self.state_num = 0
        self.function_len = []  # the state num of each functional area
        self.results = []
        self.image_length = 0
        for area in self.AreaPenalty:
            # print(math.pow(2,len(area)))
            num = math.pow(2, len(area))
            self.function_len.append(int(num))
            self.state_num += num

    def get_state_num(self) -> int:
        """
        return the state num
        :return: the num of the state
        """
        return int(self.state_num)

    def get_optimal_result(self) -> float:
        """
        get the optimal result
        :return: the optimal result
        """
        print("results' length:", len(self.results))
        return self.results[:self.image_length]

    def feature_digit(self, sign, th, feature) -> int:
        """
        implement the signal function: a sgn(x-b) + (1-a)sgn(b-x)  if -1, then 0
        :param sign: the signal
        :param th: threshold
        :param feature: feature
        :return: the result of the signal function
        """
        if sign:
            # >
            if feature > th:
                return 1
            else:
                return 0
        else:
            if feature <= th:
                return 1
            else:
                return 0

    def image_to_state(self, image, length):
        """
        mapping of the image to state
        :param image: target image
        :param length: length of the data
        :return:
        """
        w = []
        state_list = []  # Store the 0,1 list, and convert to the decimal.
        penalty_score = 0
        self.image_length = length
        if image['visual_entropy'] >= 0.6:
            # business-0
            area = self.AreaPenalty[0]
            for key in area:
                digit_ = self.feature_digit(area[key][0], area[key][1], image[key])
                state_list.append(digit_)
                penalty_score += digit_ * area[key][2]
                w.append(area[key][2])
            state = list_to_int(state_list)
            self.results.append(penalty_score)
            return w, state, state_list
        elif (image['car_num'] == 0 and image['building'] > 0.2) or (
                image['car_num'] == 0 and image['greenery'] > 0.35):
            area = self.AreaPenalty[1]
            for key in area:
                state_list.append(self.feature_digit(area[key][0], area[key][1], image[key]))
                w.append(area[key][2])
                penalty_score += self.feature_digit(area[key][0], area[key][1], image[key]) * area[key][2]
            state = list_to_int(state_list) + self.function_len[0]
            self.results.append(penalty_score)
            return w, state, state_list
        elif image['traffic_light'] == 0 and image['traffic_sign'] == 0 and image['building'] > 0.15:
            area = self.AreaPenalty[2]
            for key in area:
                state_list.append(self.feature_digit(area[key][0], area[key][1], image[key]))
                w.append(area[key][2])
                penalty_score += self.feature_digit(area[key][0], area[key][1], image[key]) * area[key][2]
            state = list_to_int(state_list) + self.function_len[0] + self.function_len[1]
            self.results.append(penalty_score)
            return w, state, state_list
        elif image['sky'] >= 0.45:
            # industry
            area = self.AreaPenalty[3]
            for key in area:
                state_list.append(self.feature_digit(area[key][0], area[key][1], image[key]))
                w.append(area[key][2])
                penalty_score += self.feature_digit(area[key][0], area[key][1], image[key]) * area[key][2]
            state = list_to_int(state_list) + self.function_len[0] + self.function_len[1] + self.function_len[2]
            self.results.append(penalty_score)
            return w, state, state_list
        elif image['traffic_sign'] == 0 and image['greenery'] >= 0.30:
            # outdoors
            area = self.AreaPenalty[4]
            for key in area:
                state_list.append(self.feature_digit(area[key][0], area[key][1], image[key]))
                w.append(area[key][2])
                penalty_score += self.feature_digit(area[key][0], area[key][1], image[key]) * area[key][2]
            state = list_to_int(state_list) + self.function_len[0] + self.function_len[1] + self.function_len[2] + \
                    self.function_len[3]
            self.results.append(penalty_score)
            return w, state, state_list

        else:
            area = self.AreaPenalty[5]
            for key in area:
                state_list.append(self.feature_digit(area[key][0], area[key][1], image[key]))
                w.append(area[key][2])
                penalty_score += self.feature_digit(area[key][0], area[key][1], image[key]) * area[key][2]
            state = list_to_int(state_list) + self.function_len[0] + self.function_len[1] + self.function_len[2] + \
                    self.function_len[3]
            self.results.append(penalty_score)
            return w, state, state_list
