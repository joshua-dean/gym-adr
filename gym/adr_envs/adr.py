import numpy as np 
import math 
from enum import Enum

class Bounds(Enum):
    LOWER = 0 
    UPPER = 1

class Direction(Enum):
    EXPAND = 0 
    SHRINK = 1

"""
Reconsider all of this
"""
class ADRVariable():
    """
    Base Automatic Domain Randomization Variable class
    For the ADR algorithm to tune a variable, it requires sampling and updating capabilities
    """
    def __init__(self):
        raise NotImplementedError
    
    def sample(self):
        raise NotImplementedError

    def boundary_sample(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

class ADRUniformDistribution(ADRVariable):
    """
    A uniform distribution to be adjusted by ADR
    """
    def __init__(self, lower, upper, delta=0.02):
        self.lower_bound = lower  
        self.upper_bound = upper 
        self.delta = delta 
    
    def sample(self):
        return np.random.uniform(self.lower_bound, self.upper_bound)
    
    def boundary_sample(self, bound):
        if bound = Bounds.LOWER:
            return self.lower_bound
        elif bound = Bounds.UPPER:
            return self.upper_bound
        raise ValueError("Unsupported bound: {}".format(bound))
    
    """
    bound is a number indicating which bound to adjust
    direction is a number indicating whether to expand or shrink the bound
    Unsupported values raise a ValueError
    """
    def update(self, bound, direction=None):
        if direction is None:
            return 
        if bound == Bounds.LOWER:
            if direction == Direction.EXPAND:
                self.lower_bound -= self.delta 
            elif direction == Direction.SHRINK:
                self.lower_bound += self.delta 
            else:
                raise ValueError("Direction value not supported: {}".format(direction))
        elif bound == Bounds.UPPER:
            if direction == Direction.EXPAND:
                self.upper_bound += self.delta 
            elif direction == Direction.SHRINK:
                self.upper_bound -= self.delta
            else:
                raise ValueError("Direction value not supported: {}".format(direction))
        else:
            raise ValueError("Bound value not supported: {}".format(bound))
    
class ADR():
    """
    class to track and update ADRVariables
    boundary_sample and update are designed to be called sequentially, with performance data collected inbetween them, i.e.:
        lam, var_info = adr.boundary_sample()
        env = construct_environment(lam)
        performance = policy.train(env)
        adr.update(performance, var_info)
    if called sequentially, var_info will be handled automatically 
    if called in a disjoint fashion, it must be passed manually

    sampling can be done without updating variables:
        lam = adr.sample()
        env = construct_environment(lam)
    """
    def __init__(self, variable_list=[], pq_size=240, performance_threshold=[0, 10]):
        self.performance_threshold = performance_threshold
        self.sample_idx = None 
        self.sample_bound = None 
        self.variables = []
        self.performance_queue = []
        self.pq_size = pq_size
        for var in variable_list:
            self.add_variable(var)

    
    def add_variable(self, var):
        if type(var) is ADRUniformDistribution:
                self.performance_queue.append([[]]*2)
            else:
                raise ValueError("Variable type not supported: {} of type {}".format(variable, type(variable)))
        self.variables.append(var)
    
    def sample(self):
        lam = [] 
        for variable in self.variables:
            lam.append(variable.sample())
        return lam 
    
    def boundary_sample(self):
        self.sample_idx = np.random.randint(0, len(self.variables))
        lam = self.sample()
        x = np.random.uniform()
        if x < 0.5:
            self.sample_bound = Bounds.LOWER 
        else:
            self.sample_bound = Bounds.UPPER 
        lam[self.sample_idx] = self.variables[self.sample_idx].boundary_sample(self.sample_bound)
        return lam, [self.sample_idx, self.sample_bound]
    
    def update(self, performance, var_info=None):
        if var_info = None:
            var_info = [self.sample_idx, self.sample_bound]
        idx = var_info[0]
        bound = var_info[1]
        self.performance_queue[idx][bound].append(performance)
        if len(self.performance_queue[idx][bound]) >= self.pq_size:
            performance_sum = 0
            for pval in self.performance_queue[idx][bound]:
                performance_sum += pval 
            self.performance_queue[idx][bound] = [] 
            performance_avg = performance_sum  / self.pq_size
            if performance_avg > self.performance_threshold[1]: #exceeds expectations
                self.variables[idx].update(bound, Direction.EXPAND)
            elif performance_avg < self.performance_threshold[0]: #underperformed
                self.variables[idx]/update(bound, Direction.SHRINK)
            else 
                pass #no modification