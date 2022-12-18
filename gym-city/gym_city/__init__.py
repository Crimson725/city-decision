from importlib_metadata import entry_points
from gym.envs.registration import register

register(
    id="city-v0",
    entry_point="gym_city.envs:City_v0",
)
register(
    id="city-v1",
    entry_point="gym_city.envs:City_v1",
)
register(
    id="city-test-v1",
    entry_point="gym_city.envs:City_Test_v1",
)
register(
    id="city-test-v3",
    entry_point="gym_city.envs:City_Test_v3",
)
register(
    id="city-v2",
    entry_point="gym_city.envs:City_v2",
)
register(
    id="city-feature-v2",
    entry_point="gym_city.envs:City_feature_v2",
)
register(
    id="city-v3",
    entry_point="gym_city.envs:City_v3",
)
register(
    id="irl-v0",
    entry_point = "gym_city.envs:IRL_World"
)
register(
    id="irl-test-v0",
    entry_point = "gym_city.envs:IRL_World_Test"
)