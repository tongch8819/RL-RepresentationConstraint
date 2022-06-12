from rcrl.envs.common.logx import get_logger

import numpy as np

class CSEnv:

    def __init__(self, 
        log_file_path = None, 
    ):

        if log_file_path is not None:
            self.logger = get_logger(log_file_path)

        self.env_name = "Abstract Cloud Simulator"
        self.current_timestep = 0
        self.max_timestep = 500

    def __str__(self):
        res =  f"=== Cloud Simulator Env ===\n"
        res +=  f"env_name: {self.env_name}\n"
        res += f"H0: {self.h0}\nREW: {self.reward_f.__name__}\n"
        res += f"UD: {self.is_dynamic}\n#U: {self.num_users}\n"
        res += f"==========================="
        return res

    def step(self, action_arr, is_random_policy=False):
        """
        Input:
        - action_arr: action with numpy.array dtype
        """
        # clip way is not good !!!
        # action_arr = np.clip(action_arr, a_min=0.0, a_max=1.0)
        # post transform could generate a nan
        # act_post_transform = lambda x : 1 / ( 1 + np.exp(-x) )
        # if not is_random_policy:
        #     trans_action_arr = act_post_transform(action_arr)
        #     if any(np.isnan(trans_action_arr)):
        #         self.logger.warning(str(action_arr))
        #         self.logger.warning(str(trans_action_arr))
        #         # raise ValueError
        #         action_arr = self.action_space.sample()
        #         self.logger.warning("Sample:" + str(action_arr))
        #     else:
        #         action_arr = trans_action_arr
        action_arr = np.clip(action_arr, a_max=1.0, a_min=0.0)

        # assert self.action_space.contains(action_arr), "Action array out of space.\n{}".format(action_arr)
        action = self.a_codec.decode_numpy(action_arr)

        # transition into next state
        next_s, is_overflow = self.trans_p.eval(self.cur_state, action)
        # compute reward
        reward, r_comps = self.reward_f(self.cur_state, action)
        # build infos
        infos = {
            'r_comps' : r_comps,
            'is_overflow' : is_overflow,
        }
        self.current_timestep += 1
        done = self.current_timestep > self.max_timestep

        next_s_arr = self.s_codec.encode_pre_transform(next_s)

        self.cur_state = next_s
        self.action_space.cur_o = next_s
        self.cur_action = action
        return next_s_arr, reward, done, infos

    def close(self):
        pass

    def seed(self, seed):
        self.seed = seed
        np.random.seed(seed)

    def split_observation(self, relax_factor : np.ndarray):
        """
        Split current state into sub-state by KNN
        Put action - relax factor into substates
        """
        neighbor_num = 3
        def find_neighbors(idx, state):
            dist = lambda x, y : abs(100 * (x - y))
            def argmin(arr):
                idx, min_v = None, None
                for i, v in enumerate(arr):
                    if idx is None:
                        idx, min_v = i, v
                    elif v < min_v:
                        idx, min_v = i, v
                return idx, min_v

            c = state.credit_vec[idx]
            idxs = []
            res = []
            for i, credit in enumerate(state.credit_vec):
                if i == idx:
                    continue
                if len(idxs) < neighbor_num:
                    idxs.append(i)
                    res.append(dist(c, credit))
                else:
                    t = dist(c, credit)
                    pop_idx, min_val = argmin(res)
                    if min_val > t:
                        res.pop(pop_idx)
                        idxs.pop(pop_idx)
                        res.append(t)
                        idxs.append(i)
            return idxs
                
        n = self.num_users
        s = self.cur_state
        obs = self.s_codec.encode_numpy(s)
        sub_obs_lst = []
        for i in range(self.num_users):
            t = np.zeros(((neighbor_num + 1) * 4 + 3, ))
            idxs = [i] + find_neighbors(i, s)
            # available quota
            t[:-3] = np.concatenate([
                s.request_vec[idxs],
                s.credit_vec[idxs],
                s.deploy_vec[idxs],
                s.allocated_vec[idxs],
            ])
            t[-3:-1] = obs[-2:]
            t[-1] = relax_factor[0]
            sub_obs_lst.append(t)
        return sub_obs_lst
        

