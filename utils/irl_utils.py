from sklearn import metrics
import pandas as pd
import numpy as np

from utils.common_utils import list_to_int, feature_digit


def save_reward(array, filename):
    """
    Save the reward to the file
    :param array: the reward array
    :param filename: name of the file to save
    :return:
    """
    length = len(array)
    f = open(filename, 'w')
    for item in array:
        f.write(str(item))
        f.write(',')
    f.close()


def cal_auc(preds, label) -> float:
    """
    calculate the auc
    :param preds: predictions
    :param label: labels
    :return: the AUC score
    """
    return metrics.roc_auc_score(label, preds)


def get_features_from_csv(path, *features) -> list:
    """
    get the features from csv
    :param path: the path of the csv file
    :param features: features need to retrieved
    :return: a list of features
    """
    df = pd.read_csv(path, encoding='utf-8')
    df = df.loc[:, features]
    feature_list = []
    for i in range(df.shape[0]):
        feature_list.append(df.loc[i].values.tolist())
    return feature_list


def get_feature_from_csv(path, feature) -> list:
    """
    get the feature from csv
    :param path: the path of the csv file
    :param feature: the name of the feature
    :return: the list of the selected feature
    """
    df = pd.read_csv(path, encoding='utf-8')
    return df[feature].tolist()


def get_image_features(path, *features) -> list:
    """
    get the list of the given features
    :param path: the path of the csv file
    :param features: features to be retrieved
    :return: the list of features
    """
    df = pd.read_csv(path)
    df = df.loc[:, features]
    feature_list = []
    for i in range(df.shape[0]):
        feature_list.append(df.loc[i].values.tolist())
    return feature_list


def cal_learned_behavior_accuracy(true_policy, learned_policy) -> float:
    """
    calculate the learned behavior accuracy
    for IRL experiment
    :param true_policy: original policy
    :param learned_policy: the rl policy learned
    :return: the accuracy
    """
    correct_count = 0
    length = len(true_policy)
    for i in range(length):
        if true_policy[i] == learned_policy[i]:
            correct_count += 1
    return correct_count / length


# ======================================= Network ===============================================
def sigmoid(x) -> float:
    """
    sigmoid function
    :param x: input
    :return: sigmoid(x)
    """
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_deriv(x) -> float:
    """
    sigmoid derivative function
    :param x: input
    :return: sigmoid derivative
    """
    return x * (1.0 - x)


def tanh(x) -> float:
    """
    tanh function
    :param x: input
    :return: tanh(x)
    """
    return np.tanh(x)


def tanh_deriv(x) -> float:
    """
    tanh derivative function
    :param x: input
    :return: tanh derivative
    """
    return 1.0 - x ** 2


def vector2action(y) -> list:
    """
    convert the vector to the action
    :param y: vector
    :return:  action list
    """
    res = []
    for item in y:
        if item[0] > item[1]:
            res.append(1)
        else:
            res.append(0)
    return res


def image2state(img) -> (int, list):
    """
    convert the image to the state
    :param img:
    :return:
    """
    # img = [] n个特征
    # 'visual_entropy', 'greenery', 'sky', 'wall', 'fence', 'sidewalk',
    #                                 'car_num', 'electric_wire'
    feature_n = len(img)  # 特征数 n
    state_list = []
    state = 0
    # 由简单计算得到
    th = [0.486313991, 0.208568784, 0.216528296, 0.01408079, 0.011293125,
          0.030639994, 2.683908046, 0.367816092]
    a = [0, 0, 0, 0, 0, 1, 1, 1]
    for i in range(feature_n):
        state_list.append(feature_digit(a[i], th[i], img[i]))
    state = list_to_int(state_list)
    # state 0-255一个数
    # state_list: [0,0,0,1,1,0,1,1] 0，1组成的列表
    return state, state_list


def int_to_list(n):
    # 从低到高[]
    res = [0] * 8
    m = str(bin(int(n)))  # 把二进制数转成字符串
    list_ = list(m[2::])
    l = list(reversed(list_))
    i = 0
    for item in l:
        s = int(item)
        res[i] = s
        i += 1

    return res
