from rcrl.envs.common.factor_action import Action, ActionCodec, ActionSpace
from rcrl.envs.common.cloud_simulator import CSEnv 
from rcrl.envs.common.single_subscriber import GoodSubscriber, BadSubscriber 
from rcrl.envs.common.config_default import abundant_init_ava_quota, deficient_init_ava_quota, get_larger_name_to_config
from rcrl.envs.common.config_default import name_to_config_exp
from rcrl.envs.common.reward_function import primary_reward_oaf, primary_reward

import numpy as np
from sklearn.preprocessing import StandardScaler

class PrimaryCSEnv(CSEnv):

    def __init__(self,
        log_dir=None,
        is_abundant=False,
        is_dynamic=False,
        with_oaf=False,
        name_to_config=name_to_config_exp):
        """
        State: full user, available quota, physical resource
        Action: full satisfaction ratio, relax factor 
        """
        
        if is_abundant:
            self.h0 = abundant_init_ava_quota 
        else:
            self.h0 = deficient_init_ava_quota
        self.is_dynamic = is_dynamic

        self.num_users = len(name_to_config)
        self.action_space = ActionSpace(self.num_users)
        self.observation_space = StateSpace(self.num_users)
        self.s_codec = StateCodec(self.num_users)

        # cannot move outside of init function
        if with_oaf:
            self.reward_f = primary_reward_oaf
        else:
            self.reward_f = primary_reward

        self.name_to_config = name_to_config
        self.a_codec = ActionCodec()
        self.env_name =  "Primary CS"

    def reset(self):
        """
        Return numpy.array
        """
        self.user_pool = self._build_user_pool()
        # bind transition probability with user pool
        self.trans_p = TransProb(self.user_pool)
        # construct init state
        cur_quota = np.array([x.cur_quota for x in self.user_pool.values()])
        s0 = State(
            request_vec        = np.array([x.send_signal() for x in self.user_pool.values()]),
            credit_vec         = np.array([x.credit_score for x in self.user_pool.values()]),
            deploy_vec         = np.array([x.deploy_vm_num for x in self.user_pool.values()]),
            allocated_vec      = cur_quota,
            satisfaction_ratio = np.random.random(size=cur_quota.size),
            a_quota            = np.array([self.h0]),
            total_core         = np.array([self.h0 + cur_quota.sum()]),
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
            if user_config['config']['behave'] == 'good':
                user_class = GoodSubscriber
            else:
                user_class = BadSubscriber
            res[name] = user_class(
                config = user_config['config'],
                credit = user_config['credit'],
                init_deploy = user_config['init_deploy'],
                cur_quota = user_config['cur_quota'],
                is_dynamic = self.is_dynamic,
            )
        return res


class PrimaryCSEnvOaf(PrimaryCSEnv):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reward_f = primary_reward_oaf



class State:

    def __init__(self,
                 request_vec: np.array,
                 credit_vec: np.array,
                 deploy_vec: np.array,
                 allocated_vec: np.array,
                 satisfaction_ratio: np.array,
                 a_quota: np.array,
                 total_core: np.array,
                 ):
        """
        Inputs:
        - request_vec: request vector r
        - credit_vec: credit vector c, 
        - allocated_vec: allocated vector x
        - a_quota: current available quota h
        """
        assert type(request_vec) is np.ndarray, TypeError
        assert type(credit_vec) is np.ndarray, TypeError
        assert credit_vec[0] is not None, TypeError
        assert type(deploy_vec) is np.ndarray, TypeError
        assert type(allocated_vec) is np.ndarray, TypeError
        assert type(satisfaction_ratio) is np.ndarray, TypeError
        assert type(a_quota) is np.ndarray, TypeError
        assert type(total_core) is np.ndarray, TypeError

        self.request_vec = request_vec
        self.credit_vec = credit_vec
        self.deploy_vec = deploy_vec
        self.allocated_vec = allocated_vec
        self.satisfaction_ratio = satisfaction_ratio
        self.available_quota = a_quota
        self.physical_resource = total_core

        self.dim = sum(map(lambda x: len(x), [
            self.request_vec,
            self.credit_vec,
            self.deploy_vec,
            self.allocated_vec,
            self.available_quota,
            self.physical_resource,
            self.satisfaction_ratio,
        ]))

    def __str__(self):
        """
        Multi-line print format.
        """
        tplt = "r:{}   c:{}  d:{}   x:{}   h:{}   w:{}   sa:{}"
        return tplt.format(
            self.request_vec,
            np.round(self.credit_vec, 2),  # round to two decimals
            self.deploy_vec,
            self.allocated_vec,
            self.available_quota,
            self.physical_resource,
            np.round(self.satisfaction_ratio, 2),
        )

    def get_user_stat(self, idx):
        t = [
            self.request_vec[idx],
            self.credit_vec[idx],
            self.deploy_vec[idx],
            self.allocated_vec[idx],
        ]
        return np.array(t)


class StateCodec:
    """
    Encode and decode State
    """

    def __init__(self, num_users):
        # , req_bnd=30, credit_box=[0.3, 0.6]):
        """
        Input:
        - num_users: number of users, used for decode functions
        - req_bnd: offset of request number, req in State lies in [-req_bnd, req_bnd]
          while req in array lies in [0, 2*req_bnd+1]
        - credit box: iterator for discretization of credit
        """
        self.num_users = num_users

    def encode_numpy(self, state):
        """
        State to np.ndarray.
        """
        return np.concatenate([
            state.request_vec,
            state.credit_vec,
            state.deploy_vec,
            state.allocated_vec,
            state.satisfaction_ratio,
            state.available_quota,
            state.physical_resource,
        ])

    def encode_pre_transform(self, state):
        """
        State to normalized np.ndarray.
        """
        def normalize(x):
            y = x.reshape((-1, 1))
            ss = StandardScaler()
            ss.fit(y)
            z = ss.transform(y)
            return z.reshape(-1)

        r = normalize(state.request_vec)
        c = normalize(state.credit_vec)
        sa = normalize(state.satisfaction_ratio)
        jumble = normalize(np.concatenate([
            state.deploy_vec,
            state.allocated_vec,
            state.available_quota,
            state.physical_resource,
        ]))
        return np.concatenate([r, c, sa, jumble])

    def decode_numpy(self, state_arr):
        """
        Np.ndarray to State object.
        """
        t = self.num_users
        return State(
            request_vec        = state_arr[:t],
            credit_vec         = state_arr[t: 2*t],
            deploy_vec         = state_arr[2*t: 3*t],
            allocated_vec      = state_arr[3*t: 4*t],
            satisfaction_ratio = state_arr[4*t: 5*t],
            a_quota            = state_arr[:-1],
            total_core         = state_arr[-1:],
        )

class StateSpace:

    def __init__(self, num_users):
        self.num_users = num_users
        self.shape = (5 * num_users + 2, )

    def contains(self, x):
        return True

class TransProb:
    """
    System Dynamics class.  next_s = f(s, a)    
    """

    def __init__(self, user_pool: dict):
        """
        Inputs:
        - user_pool: iterator of Subscriber object
        - state_codec: state codec object
        - action_codec: action codec object
        """
        # bind user pool to synchronize
        self.user_pool = user_pool

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
        # physical quota should be updated before user could adjust its deployment
        allocated_vec = state.allocated_vec + supply_vec
        allo_sum, w = allocated_vec.sum(), state.physical_resource[0]
        if allo_sum > w: 
            physi_supply = []
            for _, user in self.user_pool.items():
                x = np.floor(w * user.cur_quota / allo_sum)
                physi_supply.append(x)
        else:
            physi_supply = supply_vec

        t = []
        for user_name, supply_i, physi_supply_i in zip(self.user_pool, supply_vec, physi_supply):
            user = self.user_pool[user_name]
            signal, credit, deploy = user.step_forward(int(supply_i), int(physi_supply_i))
            t.append(user.cur_quota)
            r_prime.append(signal)
            c_prime.append(credit)
            d_prime.append(deploy)

        self._sync_check(allocated_vec, state, action, supply_vec)

        a_quota = np.clip(state.available_quota -
                          supply_vec.sum(), a_min=0, a_max=None)

        next_s = State(
            request_vec        = np.array(r_prime),
            credit_vec         = np.array(c_prime),
            deploy_vec         = np.array(d_prime),
            allocated_vec      = allocated_vec,
            satisfaction_ratio = np.zeros_like(allocated_vec), # zero padding
            a_quota            = a_quota,
            total_core         = state.physical_resource,
        )

        return next_s, is_overflow

    def _compute_supply_vec(self, state, action):
        """
        Compute allocation vector
        1. supply_vec[i] = state[i] if state[i] <= 0
        (10, 20, -1, -2) 3
        ->
        (4, 6, -1, -2) 0
        """
        # supply_vec_raw = np.floor( * )
        is_overflow = False
        req = state.request_vec
        sup = state.satisfaction_ratio
        # adjust supply vector to avoid impossible case:
        # - sum of supply is larger than available quota
        h = state.available_quota[0]
        w = state.physical_resource[0]

        neg = np.clip(req, a_max=0, a_min=None)
        allocate_bnd = h - sum(neg)

        cur_factor = state.allocated_vec.sum() / state.physical_resource - 1
        allocate_bnd += (action.relax_factor - cur_factor) * (w)

        if allocate_bnd <= 0:
            t = np.zeros_like(neg)
        else:
            t = np.floor(req * sup)
            t = np.clip(t, a_max=None, a_min=0)
            ts = t.sum()
            if ts > 0 and ts > allocate_bnd:
                assert ts > 0, f"Consistency Error\n{ts}\n{allocate_bnd}"
                # register allocation overflow
                is_overflow = True
                # equal distribute when allocation overflows
                r = t / ts
                t = np.floor(allocate_bnd * r, dtype=np.float32)
                t = np.clip(t, a_max=None, a_min=0)
        supply_vec = neg + t
        return supply_vec, is_overflow

    def _sync_check(self, allocated_vec, state, action, supply_vec):
        """synchronization consistency check"""
        delta = 10
        user_allocated_vec = np.array(
            [user.cur_quota for user in self.user_pool.values()])
        # if not all(allocated_vec == user_allocated_vec):
        # relax consistency constraint
        if np.max(allocated_vec - user_allocated_vec) > delta:
            exp_str = "\nAllocated vector is not consistent with user pool\n"
            exp_str += "State: {}\n".format(state)
            exp_str += "Action.supply_ratio_vec: {}\n".format(
                action.supply_ratio_vec)
            exp_str += "Action.supply_vec: {}\n".format(supply_vec)
            exp_str += "allocated_vec: {}\n".format(allocated_vec)
            exp_str += "user_allocated_vec: {}\n".format(user_allocated_vec)
            raise ValueError(exp_str)