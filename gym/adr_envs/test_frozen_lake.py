import torch 
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv
from frozen_lake_adr import Direction, FrozenLakeADREnv
from utils import memoized
from stable_baselines import PPO2 
from stable_baselines.common.cmd_util import make_vec_env
from adr import ADRUniform, ADRParam, ADR
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

def hard_env():
    map_size = ADRUniform(
        phi_l = ADRParam.fixed_boundary(10),
        phi_h = ADRParam.fixed_boundary(10),
        name = 'map_size'
    )
    frozen_tiles_prob = ADRUniform(
        phi_l = ADRParam.fixed_boundary(0.5),
        phi_h = ADRParam.fixed_boundary(0.5),
        name = 'tile_prob'
    )
    adr_distributions = [map_size, frozen_tiles_prob]
    env = FrozenLakeADREnv(adr_distributions=adr_distributions, do_sample=False)
    env.adr.do_boundary_sample = False
    env = make_vec_env(lambda: env, n_envs=1)

    return env

def ez_env():
    map_size = ADRUniform(
        phi_l = ADRParam.fixed_boundary(3),
        phi_h = ADRParam.fixed_boundary(3),
        name = 'map_size'
    )
    frozen_tiles_prob = ADRUniform(
        phi_l = ADRParam.fixed_boundary(0.9),
        phi_h = ADRParam.fixed_boundary(0.9),
        name = 'tile_prob'
    )
    adr_distributions = [map_size, frozen_tiles_prob]
    env = FrozenLakeADREnv(adr_distributions=adr_distributions, do_sample=False)
    env = make_vec_env(lambda: env, n_envs=1)

    return env

def mid_env():
    map_size = ADRUniform(
        phi_l = ADRParam.fixed_boundary(5),
        phi_h = ADRParam.fixed_boundary(5),
        name = 'map_size'
    )
    frozen_tiles_prob = ADRUniform(
        phi_l = ADRParam.fixed_boundary(0.7),
        phi_h = ADRParam.fixed_boundary(0.7),
        name = 'tile_prob'
    )
    adr_distributions = [map_size, frozen_tiles_prob]
    env = FrozenLakeADREnv(adr_distributions=adr_distributions, do_sample=False)
    env.adr.do_boundary_sample = False
    env = make_vec_env(lambda: env, n_envs=1)

    return env

def normal_adr_env():
    env = FrozenLakeADREnv()
    env = make_vec_env(lambda: env, n_envs=1)

    return env 

def train_policy(env, learn_steps=10000):
    model = PPO2('MlpLstmPolicy', env, nminibatches=1, verbose=1) #nminibatches=1 necessary until ADR is thread-safe for vec_envs
    model = model.learn(learn_steps)

    return model 

def eval_model(model, env, attempts):
    avg_cum_rew = 0

    obs = env.reset()

    state = None 
    done = [False]
    for i in range(attempts):
        while True:
            action, state = model.predict(obs, state=state, mask=done)
            cum_rew = env.envs[0].cum_rew 

            obs, rew, done, info = env.step(action)

            if done: 
                break 
        print(rew)
        avg_cum_rew += (cum_rew + rew) #must include last step
        env.reset()
    
    return avg_cum_rew / attempts 

def full_eval():
    learn_steps = 1
    eval_attempts = 1

    train_envs = {
        "Easy": ez_env(),
        "Medium": mid_env(),
        "Hard": hard_env(),
        "ADR": normal_adr_env()
    }
    eval_envs = {
        "Easy": ez_env(),
        "Medium": mid_env(),
        "Hard": hard_env(),
    }

    for train_key, train_env in train_envs.items():
        model = train_policy(train_env, learn_steps)

        for eval_key, eval_env in eval_envs.items():
            score = eval_model(model, eval_env, eval_attempts)

            print("{} env evaluated in {} env: {}".format(train_key, eval_key, score))

    


if __name__ == "__main__":
    full_eval()
        
    
