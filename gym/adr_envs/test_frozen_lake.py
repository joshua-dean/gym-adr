import torch 
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv
from frozen_lake_adr import Direction, FrozenLakeADREnv
from utils import memoized
import random 
import time 

def test_original_env():
    env = FrozenLakeEnv(is_slippery=True)
    obs = env.reset()
    act = Direction.RIGHT
    for _ in range(100):
        env.render()
        obs, rew, done, info = env.step(act)

# This is a questionable and non-comprehensive way to measure performance
# Notes: For many steps or environment reuse, memoizing sees anwhere from 1-5% performance increase
# For low step counts and/or single use environments this ends up being slower
# Conclusion: Memoize only if an environment instance will persist for a large amount of time (>10 runs / >1000 timesteps)
# This may be unlikely in the ADR case
# Only tested for window size of 2 (3x3)
def test_obs_transform_mem_speed():
    """
    Transition function uses ints but our observation needs a window view of the map based on position
    This tests effectiveness of memoizing the function that produces that observation.
    """
    n_runs = 100
    n_steps = 1000
    actions = [random.choice(list(Direction)) for _ in range(n_steps)]

    env = FrozenLakeADREnv(None, map_name="4x4")
    normal_start = time.time()
    for _ in range(n_runs):
        obs = env.reset()
        for idx in range(n_steps):
            act = actions[idx]
            obs, rew, done, info = env.step(act)
    normal_end = time.time()

    env.expand_obs = memoized(env.expand_obs)
    mem_start = time.time()
    for _ in range(n_runs):
        obs = env.reset()
        for idx in range(n_steps):
            act = actions[idx]
            obs, rew, done, info = env.step(act)
    mem_end = time.time()

    print("Vanilla : {}s".format(normal_end - normal_start))
    print("Memoized : {}s".format(mem_end - mem_start))

def test_DR():
    env = FrozenLakeADREnv()
    obs = env.reset()
    act = Direction.RIGHT 
    for _ in range(3):
        env.render()
        obs, rew, done, info = env.step(act)


if __name__ == "__main__":
    test_DR()
        
    
