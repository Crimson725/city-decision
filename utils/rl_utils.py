import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy import spatial
import scipy.stats


# ----------------------------TOOLS---------------------------------
def get_reward(filename) -> list:
    """
    get the reward from the file
    :param filename: name of the file to load
    :return: list of the reward
    """
    f = open(filename, 'r')
    res = f.read()
    res_list = []
    for i in range(100):
        res_list.append(float(res.split(',')[i]))
    print(res_list)
    f.close()
    return res_list


def draw_reward():
    """
    Draw the reward curve
    :return:
    """
    # f = open('reward-dqn.txt', 'r') #读
    dqn_r = get_reward('../reward-log/reward-dqn.txt')
    ppo_r = get_reward('../reward-log/reward-ppo.txt')
    sac_r = get_reward('../reward-log/reward-sac.txt')
    expert_r = get_reward('../reward-log/reward-expert.txt')
    step = []
    for i in range(100):
        step.append(i)
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False  # display the negative sign

    #    plt.bar(step,cost, color="red")
    #    plt.plot(step,cost)
    plt.plot(step, dqn_r, color="red", label="IRL+D3QN")
    plt.plot(step, ppo_r, color="blue", label="IRL+PPO")
    plt.plot(step, sac_r, color="green", label="IRL+SAC")
    plt.plot(step, expert_r, color="orange", label="Expert Reward+D3QN")

    plt.legend()
    plt.xlabel("episode number")
    plt.ylabel("mean reward")
    plt.title("The mean rewards in learning process")
    plt.show()
    plt.savefig("reward.png")


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


# def index_to_feature(feature_list, index) -> float:
#     """
#     get the feature from the index
#     :param feature_list: the list of features
#     :param index: the index of the feature
#     :return: the selected feature
#     """
#     return feature_list[index]


def get_images_feature_dict_from_csv(path) -> dict:
    """
    get the feature dict of the images from csv
    :param path: the path of the csv file
    :return: the feature dict of the images
    """
    feature = ['file_name', 'is_safe', 'greenery', 'wall', 'traffic_light', 'traffic_sign', 'isline', 'sky', 'building',
               'person', 'visual_entropy', 'vegetation', 'car_num', 'sidewalk']
    df = pd.read_csv(path, encoding='utf-8')
    df = df.loc[:, feature]
    dict_ = df.to_dict(orient='records')
    return dict_


# ====================================== Evaluation ===============================================

# Learned behavior accuracy
def cal_CC(preds, label) -> float:
    """
    calculate the correlation coefficient
    :param preds: predictions
    :param label: labels
    :return: the correlation coefficient
    """
    return pearsonr(preds, label)[0]


def cal_sim(preds, label) -> float:
    """
    calculate the cosine similarity
    :param preds: predictions
    :param label: labels
    :return: the cosine similarity
    """
    cos_sim = 1 - spatial.distance.cosine(preds, label)
    return cos_sim


def cal_kl(preds, label) -> float:
    """
    calculate the kl divergence
    :param preds: predictions
    :param label: labels
    :return: the kl divergence
    """
    preds = del_neg(preds)
    label = del_neg(label)
    return scipy.stats.entropy(preds, label)


def del_neg(l) -> list:
    """
    delete the negative value in the list
    :param l: value list
    :return: list without negative values
    """
    if min(l) < 0:
        margin = - min(l)
        for i in range(len(l)):
            l[i] = l[i] + margin + 1
    # print("a :", a)
    return l
