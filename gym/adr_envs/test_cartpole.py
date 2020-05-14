from cartpole_adr import CartPoleADREnv 
from stable_baselines import PPO2 
from stable_baselines.common.cmd_util import make_vec_env 
from adr import ADRUniform, ADRParam, ADR

def ez_env():
    grav = ADRUniform.fixed_value(
        value = 8.0,
        name = "gravity"
    )
    m_c = ADRUniform.fixed_value(
        value = 0.8,
        name = "cart_mass"
    )
    m_p = ADRUniform.fixed_value(
        value = 0.1,
        name = "pole_mass"
    )
    p_len = ADRUniform.fixed_value(
        value = 0.6,
        name = "pole_length"
    )
    force_mag = ADRUniform.fixed_value( #what's an "easy" value here? who knows
        value = 10.0,
        name = "force_magnitude"
    )
    tau = ADRUniform.fixed_value(
        value = 0.01, #faster update than ADR will see during training
        name = "tau"
    )
    distributions = [grav, m_c, m_p, p_len, force_mag, tau]
    adr = ADR(distributions, p_thresh=[0.5, 0.5])
    adr.do_boundary_sample = False 

    env = CartPoleADREnv(adr=adr)
    env = make_vec_env(lambda: env, n_envs=1)

    return env

def mid_env():
    grav = ADRUniform.fixed_value(
        value = 9.8,
        name = "gravity"
    )
    m_c = ADRUniform.fixed_value(
        value = 1.0,
        name = "cart_mass"
    )
    m_p = ADRUniform.fixed_value(
        value = 0.1,
        name = "pole_mass"
    )
    p_len = ADRUniform.fixed_value(
        value = 0.5,
        name = "pole_length"
    )
    force_mag = ADRUniform.fixed_value(
        value = 10.0,
        name = "force_magnitude"
    )
    tau = ADRUniform.fixed_value(
        value = 0.04,
        name = "tau"
    )
    distributions = [grav, m_c, m_p, p_len, force_mag, tau]
    adr = ADR(distributions, p_thresh=[0.5, 0.5])
    adr.do_boundary_sample = False 

    env = CartPoleADREnv(adr=adr)
    env = make_vec_env(lambda: env, n_envs=1)

    return env

def hard_env():
    grav = ADRUniform.fixed_value(
        value = 11.0,
        name = "gravity"
    )
    m_c = ADRUniform.fixed_value(
        value = 1.2,
        name = "cart_mass"
    )
    m_p = ADRUniform.fixed_value(
        value = 0.1,
        name = "pole_mass"
    )
    p_len = ADRUniform.fixed_value(
        value = 0.4,
        name = "pole_length"
    )
    force_mag = ADRUniform.fixed_value(
        value = 10.0,
        name = "force_magnitude"
    )
    tau = ADRUniform.fixed_value(
        value = 0.10,
        name = "tau"
    )
    distributions = [grav, m_c, m_p, p_len, force_mag, tau]
    adr = ADR(distributions, p_thresh=[0.5, 0.5])
    adr.do_boundary_sample = False 

    env = CartPoleADREnv(adr=adr)
    env = make_vec_env(lambda: env, n_envs=1)

    return env

def normal_adr_env():
    env = CartPoleADREnv()
    env = make_vec_env(lambda: env, n_envs=1)

    return env 

def resample_adr_env():
    env = CartPoleADREnv(adaptive_resampling=True)
    env = make_vec_env(lambda: env, n_envs=1)

    return env


def train_policy(env, learn_steps=10000):
    model = PPO2('MlpLstmPolicy', env, nminibatches=1, verbose=0) #nminibatches=1 necessary until ADR is thread-safe for vec_envs
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
        avg_cum_rew += (cum_rew + rew) #must include last step
        env.reset()
    
    return avg_cum_rew / attempts 

def single_eval():
    learn_steps = 10000 
    eval_attempts = 10 

    train_env = resample_adr_env()

    eval_envs = {
        "Easy": ez_env(),
        "Medium": mid_env(),
        "Hard": hard_env(),
    }

    model = train_policy(train_env, learn_steps)

    for eval_key, eval_env in eval_envs.items():
        score = eval_model(model, eval_env, eval_attempts)

        print("{} env evaluated in {} env: {}".format("ADR-R", eval_key, score))

def full_eval():
    learn_steps = 50000
    eval_attempts = 10

    train_envs = {
        "Easy": ez_env(),
        "Medium": mid_env(),
        "Hard": hard_env(),
        "ADR": normal_adr_env(),
        "ADR-R": resample_adr_env()
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
    # single_eval()