from rcrl.envs.common.single_subscriber import GreedySubscriber
from rcrl.envs.naive_cs_env import NaiveCSEnv, TransProb, State

import numpy as np

class NaiveCSEnvGreedy(NaiveCSEnv):
    def reset(self):
        """
        Return numpy.array
        """
        self.user_pool = self._build_user_pool()
        # bind transition probability with user pool
        self.trans_p = TransProbGreedy(self.user_pool)
        # construct init state
        cur_quota = np.array([x.cur_quota for x in self.user_pool.values()])
        s0 = State(
            request_vec   = np.array([x.send_signal() for x in self.user_pool.values()]),
            credit_vec    = np.array([x.credit_score for x in self.user_pool.values()]),
            deploy_vec    = np.array([x.deploy_vm_num for x in self.user_pool.values()]),
            allocated_vec = cur_quota,
            a_quota       = np.array([self.h0]),
            total_core    = np.array([self.h0 + cur_quota.sum()]),
        )
        self.total_vm_cores = (s0.allocated_vec.sum() + s0.available_quota)[0]
        s0_arr = self.s_codec.encode_pre_transform(s0)

        self.cur_state = s0
        self.action_space.cur_o = s0

        # s0_arr is the input of network, so we could transform it. 
        # Under this setting, the cur_state and s_arr may look like inconsistent, but they 
        # are actually not.
        return s0_arr

    def _build_user_pool(self):
        res = dict()
        for name, user_config in self.name_to_config.items():
            res[name] = GreedySubscriber(
                config = user_config['config'],
                credit = user_config['credit'],
                init_deploy = user_config['init_deploy'],
                cur_quota = user_config['cur_quota'],
                is_dynamic = self.is_dynamic,
            )
        return res

class TransProbGreedy(TransProb):
    """
    System Dynamics class.  next_s = f(s, a)    
    """

    def eval(self, state, action):
        """
        Transit from state action into next state
        Object Version

        inputs:
        state: State object
        action: Action object
        kwargs:
            send_signal_method: naive, two_stage
            adjust_deploy_method: naive
            gen_credit_method: naive, predictive 
        """

        # container for generated request signal and credit score
        r_prime, c_prime, d_prime = [], [], []
        supply_vec, is_overflow = self._compute_supply_vec(state, action)

        t = []
        for user_name, supply_i, pre_signal in zip(self.user_pool, supply_vec, self.pre_signals):
            user = self.user_pool[user_name]
            signal, credit, deploy = user.step_forward(int(supply_i), pre_signal)
            t.append(user.cur_quota)
            r_prime.append(signal)
            c_prime.append(credit)
            d_prime.append(deploy)

        allocated_vec = state.allocated_vec + supply_vec
        self._sync_check(allocated_vec, state, action, supply_vec)

        a_quota = np.clip(state.available_quota -
                          supply_vec.sum(), a_min=0, a_max=None)

        next_s = State(
            request_vec        = np.array(r_prime),
            credit_vec         = np.array(c_prime),
            deploy_vec         = np.array(d_prime),
            allocated_vec      = allocated_vec,
            a_quota            = a_quota,
            total_core         = state.physical_resource,
        )
        return next_s, is_overflow
