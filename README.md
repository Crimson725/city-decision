# README

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