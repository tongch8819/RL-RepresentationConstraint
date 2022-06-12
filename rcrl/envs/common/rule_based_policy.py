from rcrl.envs.naiveplusfactor.mdp import StateCodec, Action

import numpy as np

class RuleBasedPolicy:
    """
    CRAL allocation policy class. 
    """
    def __init__(self, num_users):
        self.s_codec = StateCodec(num_users)

    def eval(self, state, method):
        """
        Negative dimension should be equal between state.request and action.

        Output:
        - action
        """
        if method == 'FCFS':
            next_a = self.FCFS(state)
        elif method == 'FD':
            next_a = self.FD(state)
        elif method == 'CB':
            next_a = self.CB(state)
        elif method == 'random':
            next_a = self.random_method(state)
        else:
            raise NotImplementedError

        r_v = state.request_vec
        s_v = next_a.supply_ratio_vec * r_v
        assert (r_v[r_v < 0] == s_v[s_v < 0]).all(), \
            "Recycle assumption broken.\n{}\n{}".format(r_v, s_v)

        return next_a

    def build_func(self, method):
        if method == 'FCFS':
            return self.FCFS
        elif method == 'FD':
            return self.FD
        elif method == 'CB':
            return self.CB
        elif method == 'random':
            return self.random_method
        else:
            raise NotImplementedError

    def random_method(self, state_arr):
        state = self.s_codec.decode_numpy(state_arr)
        n = len(state.request_vec)
        supply_ratio = np.random.random((n,))
        supply_ratio[state.request_vec < 0] = 1.0
        next_arr = supply_ratio
        # next_a = Action(supply_ratio)
        return next_arr

    def FCFS(self, state):
        """
        Fist Come First Serve.

        Cons:
        - cause large variance in allocation vector

        Return: Action object
        """
        n_users = len(state.request_vec)
        remain_quota = state.available_quota[0]
        supply_vec = np.zeros((n_users, ))
        for i, req in enumerate(state.request_vec):
            if req <= 0:
                supply_vec[i] = req
            else:
                if req <= remain_quota:
                    supply_vec[i] = req
                    remain_quota -= req
                else:
                    supply_vec[i] = remain_quota
                    break
        supply_ratio_vec = []
        for a, b in zip(supply_vec, state.request_vec):
            if b == 0.0:
                supply_ratio_vec.append(0.0)
            else:
                supply_ratio_vec.append(a / b)
        supply_ratio_vec = np.array(supply_ratio_vec)
        return Action(supply_ratio_vec) 

    def FD(self, state):
        """
        Fair distribution:
        1. select request user
        2. find the min request
        3. min the minimal request and maximal average request

        Return: Action object
        """
        n_users = len(state.request_vec)
        remain_quota = state.available_quota[0]

        t = state.request_vec[state.request_vec > 0]
        num_posi_req = len(t)
        if num_posi_req == 0:
            supply_vec = state.request_vec
            return Action(supply_vec)
        x = min(min(t), int(remain_quota / num_posi_req ))
        supply_vec = np.zeros((n_users, ))
        for i, req in enumerate(state.request_vec):
            if req > 0:
                supply_vec[i] = x
            else:
                supply_vec[i] = req
        supply_ratio_vec = np.nan_to_num(supply_vec / state.request_vec)
        return Action(supply_ratio_vec)
    
    def CB(self, state):
        """
        Credit Based Policy
        """
        n_users = len(state.request_vec)
        supply_vec = np.zeros((n_users, ))
        for i, req in enumerate(state.request_vec):
            if req > 0:
                supply_vec[i] = state.credit_vec[i]
            else:
                supply_vec[i] = 1.0
        supply_ratio_vec = supply_vec
        assert all(0.0 <= supply_ratio_vec) and  all(supply_ratio_vec <=1.0), "ValueError: \n{}\n{}".format(supply_vec, state.request_vec)
        return Action(supply_ratio_vec)
        # return Action(supply_ratio_vec, np.random.random((1,)))

