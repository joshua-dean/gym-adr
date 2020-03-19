import numpy as np 
from enum import IntEnum

"""
Currently only supports generics randomizers
Rewrite is probably needed for custom ones as different parameters from the same distribution are sampled at different intervals
"""

class Bound(IntEnum):
    LOWER, UPPER = range(2)

class Direction(IntEnum):
    EXPAND, SHRINK = range(2)

"""
Defined in Appendix B of ADR paper
"""
def g_func(x):
    return np.exp(x - 1)

class ADRParam():
    """
    Automatic Domain Randomization single parameter
    Contains a value, min-max bounds, a delta, a performance queue, and a sampling weight
    """
    def __init__(self, value, val_bound, delta=0.02, pq_size=240, boundary_sample_weight=1):
        self.value = value 
        self.val_bound = val_bound 
        self.delta = delta
        self.pq_size = pq_size 
        self.pq = [] 
        self.boundary_sample_weight = boundary_sample_weight 
        self.boundary_sample_flag = False 
    
    """
    Takes a performance value, updates it pq is full based on average performance over pq_size updates
    """
    def update(self, p_val, p_thresh):
        self.pq.append(p_val)
        if len(self.pq) >= self.pq_size:
            pq_avg = np.mean(self.pq)
            self.pq = [] 
            if pq_avg < p_thresh[Bound.LOWER]:
                self.value -= self.delta 
            elif pq_avg > p_thresh[Bound.UPPER]:
                self.value += self.delta
            
            self.value = np.clip(self.value, self.val_bound[Bound.LOWER], self.val_bound[Bound.UPPER])
    
    def get_boundary_sample_flag(self):
        return self.boundary_sample_flag

    def set_boundary_sample_flag(self, flag_val):
        self.boundary_sample_flag = flag_val
    
    def get_boundary_sample_weight(self):
        return self.boundary_sample_weight

    """
    Returns value
    """
    def get_value(self):
        return self.value 
    

class ADRUniform():
    def __init__(self, phi_left: ADRParam, phi_right: ADRParam):
        self.phi_left = phi_left 
        self.phi_right = phi_right
        self.last_sample = None
        self.parameters = [self.phi_left, self.phi_right]
    
    def get_parameters(self):
        return self.parameters 

    def sample(self):
        for param in parameters:
            if param.get_boundary_sample_flag():
                param.set_boundary_sample_flag(False)
                self.last_sample = param.get_value()
        self.last_sample = np.random.uniform(self.phi_left.get_value(), self.phi_right.get_value())
        return self.last_sample #return for convienience
    
    """
    Provided if Uniform is used as the desired distribution
    """
    def episode_sample(self):
        return self.sample()
    
    def get_last_sample(self):
        return self.last_sample

class ADRAdditiveGaussian():
    def __init__(self, x_0, lam_i: ADRUniform, lam_j: ADRUniform, alpha=0.01):
        self.x_0 = x_0
        self.lam_i = lam_i 
        self.lam_j = lam_j 
        self.alpha
        self.parameters = [self.lam_i.get_parameters() + self.lam_j.get_parameters]
    
    def get_parameters(self):
        return self.parameters
    
    def episode_sample(self):
        return self.x_0 + np.abs(
            np.random.normal(
                g_func(
                    self.alpha * self.lam_i.sample()
                    ),
                g_func(
                    np.abs(
                        self.alpha * self.lam_j.sample()
                    )
                )
            )
        )

class ADRUnbiasedAdditiveGaussian():
    def __init__(self, x_0, lam_i: ADRUniform, alpha=0.01):
        self.x_0 = x_0 
        self.lam_i = lam_i 
        self.alpha = alpha 
        self.parameters = self.lam_i.get_parameters() 
    
    def get_parameters(self):
        return self.parameters 

    def episode_sample(self):
        return self.x_0 + np.random.normal(
            0,
            g_func(
                np.abs(
                    self.alpha * self.lam_i.sample()
                )
            )
        )

class ADRMultiplicative():
    def __init__(self, x_0, lam_i: ADRUniform, lam_j: ADRUniform, alpha=0.01):
        self.x_0 = x_0 
        self.lam_i = lam_i 
        self.lam_j = lam_j 
        self.alpha = alpha 
        self.parameters = self.lam_i.get_parameters() + self.lam_j.get_parameters 
    
    def get_parameters(self):
        return self.parameters 
    
    def episode_sample(self):
        return self.x_0 * np.exp(
            np.random.normal(
                self.alpha * self.lam_i.sample(),
                np.abs(
                    self.alpha * self.lam_j.sample()
                )
            )
        )

class ADRActionNoise():
    def __init__(self, a_0, lam_i: ADRUniform, lam_j: ADRUniform, lam_k: ADRUniform):
        self.a_0 = a_0 
        self.lam_i = lam_i 
        self.lam_j = lam_j 
        self.lam_k = lam_k 
        self.parameters = self.lam_i.get_parameters() + self.lam_j.get_parameters + self.lam_k.get_parameters
    
    def get_parameters(self):
        return self.parameters 

    def episode_sample(self):
        self.lam_i.sample() 
        self.lam_j.sample()
        return self.step_sample()
    
    def step_sample(self):
        return a_0 * np.random.normal(
            1,
            g_func(
                np.abs(
                    self.lam_i.get_last_sample()
                )
            )
        ) 
        + np.random.normal(
            0,
            g_func(
                np.abs(
                    self.lam_j.get_last_sample()
                )
            )
        )
        + np.random.normal(
            0,
            g_func(
                np.abs(
                    self.lam_k.sample()
                )
            )
        )


class ADR():

    def __init__(self, distributions=[], p_thresh=[0, 10]):
        self.distributions = distributions 
        self.p_thresh = p_thresh 
        self.parameters = []
        self.sample_idx = None 
        for dist in self.distributions:
            self.parameters += dist.get_parameters()

    def episode_sample(self):
        lam = [] 
        for dist in self.distributions:
            lam.append(dist.episode_sample())
        return lam 
    
    def boundary_sample(self):
        sample_weights = [param.get_boundary_sample_weight() for param in self.parameters]
        weights_norm = sample_weights / np.sum(sample_weights)
        self.sample_idx = np.random.choice(len(self.parameters), p=weights_norm)
        self.parameters[self.sample_idx].set_boundary_sample_flag(True)

        return self.sample(), self.sample_idx
    
    def update(self, performance, param_idx=None):
        if param_idx is None:
            param_idx = self.sample_idx 
        self.parameters[param_idx].update(performance, self.p_thresh)

