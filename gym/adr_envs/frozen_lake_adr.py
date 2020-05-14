import sys 
from contextlib import closing 

import numpy as np 
from six import StringIO, b 

from gym import utils, spaces
from gym.envs.toy_text import discrete 

from enum import IntEnum

import adr #TODO fix imports like this
from adr import ADR, ADRParam, ADRUniform
from gym.envs.toy_text.frozen_lake import MAPS, generate_random_map

class Direction(IntEnum):
    LEFT, DOWN, RIGHT, UP = range(4)

"""
I omit sliperiness and only randomize map size and the probability a map square is frozen i.e. walkable
"""
class FrozenLakeADREnv(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, adr_distributions=None, do_sample=True, adaptive_resampling=False, desc=None, map_name="4x4"):
        """
        adr_distributions as a list of ADRDist
        """
        self.adaptive_resampling = adaptive_resampling

        if adr_distributions is None:
            map_size = ADRUniform(
                phi_l = ADRParam.fixed_boundary(3),
                phi_h = ADRParam(
                    value = 3,
                    val_bound = [3, 10],
                    delta = 1,
                    pq_size = 25
                ),
                name = 'map_size'
            )
            frozen_tiles_prob = ADRUniform(
                phi_l = ADRParam(
                    value = 0.8,
                    val_bound = [0.5, 0.8],
                    delta = -0.05,
                    pq_size = 50
                ),
                phi_h = ADRParam.fixed_boundary(1.0),
                name = 'tile_prob'
            )
            adr_distributions = [map_size, frozen_tiles_prob]

        self.adr = ADR(adr_distributions, p_thresh=[-3, 3])
        self.adr.do_boundary_sample = do_sample #ugh I hate this
        self.sample_params()
        self.generate_map()
        self.reward_range = (0, 1)
        self.currently_sampling = False
        nS, nA, P, isd = self.generate_states_actions_transitions()

        super(FrozenLakeADREnv, self).__init__(nS, nA, P, isd)

        self.action_space = spaces.Discrete(self.nA)
        # self.observation_spaces = spaces.MultiDiscrete(

        # )
        self.observation_space = spaces.Box(
            low = 0,
            high = 4,
            shape=(3,3,),
            dtype=np.float32
        )
        self.cum_rew = 0
    
    def sample_params(self):
        base_vals = [3, 0.8]
        
        while True:
            # Sampling
            if np.random.rand() > 0.5: #randomly boundary sample
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
            if width < (0.5 * self.adr.total_distribution_width()): #we force resampling if we're not utilizing 50% of the total available difficulty
                continue 
            else:
                break

        self.map_size, self.frozen_tiles_prob = lam 
        self.map_size = int(self.map_size)
    
    def generate_map(self):
        desc = generate_random_map(self.map_size, self.frozen_tiles_prob)
        self.desc = desc = np.asarray(desc, dtype='c')
        self.nrow, self.ncol = desc.shape
        self.padded_desc = np.pad(self.desc, 1, mode='constant', constant_values=' ') # is space character an bad idea? who knows
    
    @staticmethod
    def rew_by_char(char):
        char_map = {
            b'F': -0.1,
            b'H': -10.0,
            b'S': -0.1,
            b'G': 100.0,
        }
        return char_map.get(char)

    def generate_states_actions_transitions(self):
        nA = 4 
        nS = self.nrow * self.ncol 
        isd = np.array(self.desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        nrow = self.nrow 
        ncol = self.ncol

        P = {s : {a : [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row*ncol + col

        def inc(row, col, a):
            if a == Direction.LEFT:
                col = max(col-1,0)
            elif a == Direction.DOWN:
                row = min(row+1,nrow-1)
            elif a == Direction.RIGHT:
                col = min(col+1,ncol-1)
            elif a == Direction.UP:
                row = max(row-1,0)
            return (row, col)
        
        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = P[s][a]
                    letter = self.desc[row, col]
                    if letter in b'GH':
                        rew = FrozenLakeADREnv.rew_by_char(newletter)
                        li.append((1.0, s, rew, True))
                    else:
                        newrow, newcol = inc(row, col, a)
                        newstate = to_s(newrow, newcol)
                        newletter = self.desc[newrow, newcol]
                        done = bytes(newletter) in b'GH'
                        # rew = float(newletter == b'G')
                        rew = FrozenLakeADREnv.rew_by_char(newletter)
                        li.append((1.0, newstate, rew, done))
        
        return nS, nA, P, isd

    @staticmethod
    def bit_to_int(bit):
        bit_map = {
            b' ': 0,
            b'F': 1,
            b'H': 2,
            b'S': 3,
            b'G': 4,
        }
        return bit_map.get(bit)


    def expand_obs(self, obs):
        r_idx = int(obs / self.ncol) 
        c_idx = obs % self.nrow

        r_idx += 1
        c_idx += 1

        view_radius = 2

        # This pulls out a "window" observation around the current location
        ret_obs = self.padded_desc[
            r_idx - view_radius + 1:r_idx + view_radius, 
            c_idx - view_radius + 1:c_idx + view_radius
        ]
        float_obs = np.ndarray((3,3), dtype=np.float32)
        for row in range(3):
            for col in range(3):
                float_obs[row][col] = FrozenLakeADREnv.bit_to_int(ret_obs[row][col]) #doing this conversion every time is probably slow


        return float_obs
    
    def reset(self):
        self.current_steps = 0
        # print("Max Map Size : {} Min Tile Prob : {}".format(
        #     self.adr.distribution_dict['map_size'].phi_h.value, 
        #     self.adr.distribution_dict['tile_prob'].phi_l.value
        # ))
        if self.currently_sampling:
            self.adr.update(self.cum_rew)
            self.currently_sampling = False
        self.cum_rew = 0
        self.sample_params()
        self.generate_map()
        nS, nA, P, isd = self.generate_states_actions_transitions()
        self.P = P 
        self.isd = isd 
        self.nS = nS 
        self.nA = nA 
        
        obs = super().reset()
        obs = self.expand_obs(obs)

        return obs

    def step(self, a):
        obs, rew, done, info = super().step(a)
        obs = self.expand_obs(obs)
        self.cum_rew += rew

        self.current_steps += 1
        if self.current_steps > self.nS * 2: #they've had enough time to traverse every space twice
            return (obs, -20, True, info)

        return (obs, rew, done, info)

    
    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Left","Down","Right","Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()
