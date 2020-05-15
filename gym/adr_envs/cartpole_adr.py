import math 
import gym 
from gym import spaces 
from gym.utils import seeding 
from gym.envs.classic_control import CartPoleEnv
import numpy as np

import adr 
from adr import ADR, ADRParam, ADRUniform 

class CartPoleADREnv(CartPoleEnv):

    def __init__(self, adr=None, adaptive_resampling=False):
        super(CartPoleADREnv, self).__init__()

        if adr is None:
            grav = ADRUniform.centered_around(
                low = 8.0,
                start = 9.8,
                high = 11.0,
                delta = 0.02,
                pq_size = 25,
                name = "gravity"
            )
            m_c = ADRUniform.centered_around(
                low = 0.8,
                start = 1.0,
                high = 1.2,
                delta = 0.05,
                pq_size = 25,
                name = "cart_mass"
            )
            m_p = ADRUniform.fixed_value(
                value = 0.1,
                name = "pole_mass"
            )
            p_len = ADRUniform.centered_around(
                low = 0.4,
                start = 0.5,
                high = 0.6,
                delta = 0.05,
                pq_size = 25,
                name = "pole_length"
            )
            force_mag = ADRUniform.centered_around(
                low = 8.0, 
                start = 10.0, 
                high = 12.0,
                delta = 0.5,
                pq_size = 25,
                name = "force_magnitude"
            )
            tau = ADRUniform(
                phi_l = ADRParam.fixed_boundary(0.02),
                phi_h = ADRParam(
                    value = 0.02,
                    val_bound = [0.02, 0.10],
                    delta = 0.01,
                    pq_size = 25,
                    boundary_sample_weight = 2,
                ),
                name = "tau"
            )

            distributions = [grav, m_c, m_p, p_len, force_mag, tau]
            self.adr = ADR(distributions, p_thresh=[0.5, 0.5])
            self.adr.do_boundary_sample = True 
        else:
            self.adr = adr   
        self.adaptive_resampling = adaptive_resampling

        self.sample_params() 
        self.currently_sampling = False 
        self.cum_rew = 0


    def sample_params(self):
        base_vals = [9.8, 1.0, 0.1, 0.5, 10.0, 0.02]
        while True:
            # Sampling 
            if np.random.rand() > 0.5:
                self.currently_sampling = True 
                lam, _ = self.adr.boundary_sample() 
            else:
                self.currently_sampling = False 
                lam = self.adr.episode_sample() 
            
            if not self.adaptive_resampling:
                break 
                
            #evaluate sample 
            width = 0 
            for i in range(len(lam)):
                width += abs(base_vals[i] - lam[i])
            if width < (0.3 * self.adr.total_distribution_width()):
                continue 
            else:
                break 
        
        grav, m_c, m_p, pole_len, force_mag, tau = lam

        self.gravity = grav 
        self.masscart = m_c 
        self.masspole = m_p 
        self.total_mass = (self.masspole + self.masscart)
        self.length = pole_len 
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = force_mag 
        self.tau = tau 
    
    def reset(self):
        if self.currently_sampling:
            self.adr.update(self.cum_rew)
            self.currently_sampling = False 
        
        self.sample_params()
        self.cum_rew = 0

        obs = super().reset() 

        return obs 
    
    def step(self, a):
        obs, rew, done, info = super().step(a)
        self.cum_rew += rew 

        return (obs, rew, done, info)

