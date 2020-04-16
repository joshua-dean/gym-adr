import sys 
from contextlib import closing 

import numpy as np 
from six import StringIO, b 

from gym import utils 
from gym.envs.toy_text import discrete 

from enum import IntEnum

import adr #TODO fix imports like this
from gym.envs.toy_text.frozen_lake import MAPS, generate_random_map

class Direction(IntEnum):
    LEFT, DOWN, RIGHT, UP = range(4)

class FrozenLakeADREnv(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, adr_params, desc=None, map_name="4x4", is_slippery=True):
        if desc is None and map_name is None:
            desc = generate_random_map()
        elif desc is None:
            desc = MAPS[map_name]

        self.desc = desc = np.asarray(desc, dtype='c')
        self.padded_desc = np.pad(self.desc, 1, mode='constant', constant_values=' ') # is space character an bad idea? who knows
        self.nrow, self.ncol = nrow,ncol = desc.shape 
        self.reward_range = (0, 1)

        nA = 4 
        nS = nrow * ncol 

        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        # P = {s : {a : [] for a in range(nA)} for s in range(nS)}
        P = {s : {a : [] for a in Direction} for s in range(nS)}

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
                    letter = desc[row, col]
                    if letter in b'GH':
                        li.append((1.0, s, 0, True))
                    else:
                        if is_slippery:
                            for b in [(a-1)%4, a, (a+1)%4]:
                                newrow, newcol = inc(row, col, b)
                                newstate = to_s(newrow, newcol)
                                newletter = desc[newrow, newcol]
                                done = bytes(newletter) in b'GH'
                                rew = float(newletter == b'G')
                                li.append((1.0/3.0, newstate, rew, done))
                        else:
                            newrow, newcol = inc(row, col, a)
                            newstate = to_s(newrow, newcol)
                            newletter = desc[newrow, newcol]
                            done = bytes(newletter) in b'GH'
                            rew = float(newletter == b'G')
                            li.append((1.0, newstate, rew, done))

        super(FrozenLakeADREnv, self).__init__(nS, nA, P, isd)
    
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

        return ret_obs
    
    def reset(self):
        obs = super().reset()
        obs = self.expand_obs(obs)

        return obs

    def step(self, a):
        obs, rew, done, info = super().step(a)
        obs = self.expand_obs(obs)

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
