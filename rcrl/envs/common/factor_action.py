import numpy as np

class Action:
    """
    Extened Action with two main fields
    - supply vector 
    - relax factor
    """

    def __init__(self, relax_factor : np.array):
        """
        Inputs:
        - supply_ratio_vec: supply ratio b
        """
        self.relax_factor = relax_factor

    def __str__(self):
        tplt = "C:{}"
        return tplt.format(self.relax_factor)


class ActionCodec:
    """
    Encode and decode Action 
    """
    def encode_numpy(self, action):
        """
        To numpy array
        """
        return action.relax_factor

    def decode_numpy(self, action_np):
        """
        To Action
        """
        return Action(relax_factor=np.array([action_np[-1]]))


class ActionSpace:

    def __init__(self, num_users):
        self.shape = (1, )
        # self.low = (0.0, )
        # self.high = (1.0, )
        # act_limit experiment
        self.low = (0.1, )
        self.high = (0.9, )

        self.cur_o = None

    def sample(self):
        """
        Return numpy.ndarray
        """
        return np.random.random(size=(1,))

    def contains(self, x):
        x = np.array(x)
        return (
            x.shape == self.shape
            and np.all(x >= self.low[0])
            and np.all(x <= self.high[0])
        )

