import numpy as np

class Action:
    """
    Extened Action with two main fields
    - supply vector 
    - relax factor
    """

    def __init__(self, supply_ratio_vec : np.array, relax_factor : np.array):
        """
        Inputs:
        - supply_ratio_vec: supply ratio b
        """
        assert type(supply_ratio_vec) is np.ndarray, TypeError
        assert all(0.0 <= supply_ratio_vec) and all(supply_ratio_vec <= 1.0), "ValueError: \n{}".format(supply_ratio_vec)
        self.supply_ratio_vec = supply_ratio_vec
        self.relax_factor = relax_factor

    def __str__(self):
        tplt = "b: {}   C:{}"
        return tplt.format(self.supply_ratio_vec, self.relax_factor)


class ActionCodec:
    """
    Encode and decode Action 
    """
    def encode_numpy(self, action):
        """
        To numpy array
        """
        return np.concatenate([action.supply_ratio_vec, action.relax_factor])  

    def decode_numpy(self, action_np):
        """
        To Action
        """
        return Action(np.array(action_np[:-1]), np.array([action_np[-1]]))


class ActionSpace:

    def __init__(self, num_users):
        self.shape = (num_users + 1, )
        # self.low = (0.0, )
        # self.high = (1.0, )
        # act_limit experiment
        self.low = (0.1, )
        self.high = (0.9, )

        self.cur_o = None

    def sample(self):
        if self.cur_o is None:
            raise ValueError
        return self._sample(self.cur_o)

    def _sample(self, state):
        """Conditional Sample"""
        t = np.random.uniform(self.low, self.high, self.shape[0]-1)
        no_action_mask = (state.request_vec == 0)
        recycle_mask = (state.request_vec < 0)
        t[no_action_mask] = 0.0
        t[recycle_mask] = 1.0

        r = np.random.uniform(self.low, self.high, 1)
        return np.concatenate([t, r])

    def contains(self, x):
        x = np.array(x)
        return (
            x.shape == self.shape
            and np.all(x >= self.low[0])
            and np.all(x <= self.high[0])
        )

    def constant(self):
        t = np.ones(self.shape)
        return t

