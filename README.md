# README

This repository provides a python implementation of the RL framework of our paper [**A Complete Reinforcement-Learning-Based Framework for Urban-Safety Perception**](https://www.mdpi.com/2220-9964/11/9/465).

## Citation and Contact

You can find a PDF of the paper at 

If you use our work, please also cite the paper:

```
@article{wang2022complete,
  title={A Complete Reinforcement-Learning-Based Framework for Urban-Safety Perception},
  author={Wang, Yaxuan and Zeng, Zhixin and Li, Qiushan and Deng, Yingrui},
  journal={ISPRS International Journal of Geo-Information},
  volume={11},
  number={9},
  pages={465},
  year={2022},
  publisher={MDPI}
}
```

## Abstract

Urban-safety perception is crucial for urban planning and pedestrian street preference studies. With the development of deep learning and the availability of high-resolution street images, the use of artificial intelligence methods to deal with urban-safety perception has been considered adequate by many researchers. However, most current methods are based on the feature-extraction capability of convolutional neural networks (CNNs) with large-scale annotated data for training, mainly aimed at providing a regression or classification model. There remains a lack of interpretable and complete evaluation systems for urban-safety perception. To improve the interpretability of evaluation models and achieve human-like safety perception, we proposed a complete decision-making framework based on reinforcement learning (RL). We developed a novel feature-extraction module, a scalable visual computational model based on visual semantic and functional features that could fully exploit the knowledge of domain experts. Furthermore, we designed the RL module—comprising a combination of a Markov decision process (MDP)-based street-view observation environment and an intelligent agent trained using a deep reinforcement-learning (DRL) algorithm—to achieve human-level perception abilities. Experimental results using our crowdsourced dataset showed that the framework achieved satisfactory prediction performance and excellent visual interpretability.

## Environment Setup

### Requirements

```
pip install -r requirements.txt
```

### Environment Register

Go to the directory of the environment, e.g., `/gym-city`, and run the following command to register the learning
environment.

```
pip install -e.
```

## Training & Evaluation

If you want to train and evaluate the model.

```
python train_for_only_image.py
```

### Rollout

If you only want to evaluate the model.

```
python rollout.py
```

## Some important information you may want to read

[Custom env for ray](https://medium.com/distributed-computing-with-ray/anatomy-of-a-custom-environment-for-rllib-327157f269e5)

[Tutorials on Ray and Ray-based Libraries](https://github.com/anyscale/academy/)

## Future Work

- Rewrite the codes to be compatible with the new version of Ray
- Reconstruct the environments for better readability and performance