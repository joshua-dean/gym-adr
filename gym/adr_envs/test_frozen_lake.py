import torch 
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv
from frozen_lake_adr import Direction, FrozenLakeADREnv
from utils import memoized

def test_original_env():
    env = FrozenLakeEnv(is_slippery=True)
    obs = env.reset()
    act = Direction.RIGHT
    for _ in range(100):
        env.render()
        obs, rew, done, info = env.step(act)

def test_obs_speed():
    env = FrozenLakeADREnv(None)
    for _ in range(10):
        obs = env.reset()
        act = Direction.RIGHT 
        for _ in range(100):
            env.render()
            print(obs)
            obs, rew, done, info = env.step(act)


if __name__ == "__main__":
    test_obs_speed()
        
    
