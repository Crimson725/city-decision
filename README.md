# README

## Environment Setup
### Requirements


   ```
   pip install -r requirements.txt
   ```

### Environment Register
Go to the directory of the environment, e.g., `/gym-city`, and run the following command to register the learning environment

```
pip install -e.
```

 

## Evaluation & Visualize the results 

```
tensorboard --logdir=$HOME/ray_results
```

## Reference

[Custom env for ray](https://medium.com/distributed-computing-with-ray/anatomy-of-a-custom-environment-for-rllib-327157f269e5)

[Tutorials on Ray and Ray-based Libraries](https://github.com/anyscale/academy/)

## Future Work
- Rewrite training functions to be compatible with the new version of Ray
-  Rewrite the codes using PyTorch as backend
-  Reconstruct the environments for better readability and performance