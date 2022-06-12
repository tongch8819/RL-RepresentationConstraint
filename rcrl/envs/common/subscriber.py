from copy import deepcopy
import numpy as np
import pandas as pd

class Subscriber:

    def __init__(self, config, cur_quota=0, credit=0.5, init_deploy=0, is_dynamic=False):
        """
        User Representation in a Cloud Computing System
        
        Inputs:
        - signal: initial behavior signal
        - cur_quota: current quota occupied by this user
        - score: credit score, a metric for historical behavior
        - deploy: current deploy vm core
        """
        # quota related
        self.quota_hist = [cur_quota]
        self.signal_hist = []
        self.supply_hist = []
        self.cur_quota = cur_quota
        # indicate the physical upper bound when the sum of quotas exceeds the 
        # total amount of physical resources due to relax factor
        self.cur_quota_physi = self.cur_quota

        # credit score related
        self.credit_hist = [credit]
        self.credit_hist_len = 1
        self.deploy_ratio_sum = 0
        self.credit_score = credit 

        # deployment related
        self.delta_deploy_hist = []
        self.deploy_hist = [init_deploy]
        self.deploy_vm_num = init_deploy 
        self.confined_hist = []

        # adaptive config
        self.default_config = deepcopy(config)
        self.config = config
        self.param_hist = [
            self.config['request'] + self.config['recycle']
        ]
        self.prev_mu = {
            'request' : self.config['request'][1] * self.config['request'][2],
            'recycle' : self.config['request'][1] * self.config['request'][2],
        }
        self.prev_var = {
            'request' : self.prev_mu['request'] * self.config['request'][2],
            'recycle' : self.prev_mu['recycle'] * self.config['request'][2],
        }

        self.step = 0

        self.behave= self.config['behave']
        self.is_dynamic = is_dynamic
        

    def dump_hist_data(self, save_path):
        """
        Dump historial data.
        """
        def make_param_hist():
            """
            List of dictionary
            Return a dictionary
            """
            res = {
                'req_p1' : [], 
                'req_shape' : [], 
                'req_scale' : [], 
                'rec_p3' : [], 
                'rec_shape' : [], 
                'rec_scale' : [], 
            }
            for vals in self.param_hist:
                # vals list type with length of 6
                res['req_p1'].append(vals[0])
                res['req_shape'].append(vals[1])
                res['req_scale'].append(vals[2])
                res['rec_p3'].append(vals[3])
                res['rec_shape'].append(vals[4])
                res['rec_scale'].append(vals[5])
            return res
        A = {
            'quota'   : self.quota_hist,
            'credit'  : self.credit_hist,
            'deploy'  : self.deploy_hist,
            'signal'  : self.signal_hist,   # last state has signal
            'supply'  : self.supply_hist + [None],
            'delta_deploy'  : self.delta_deploy_hist + [None],
        }
        B = make_param_hist()
        A.update(B)
        # print([len(x) for x in A.values()])
        df = pd.DataFrame(A)
        df.to_csv(save_path)
        return df

    def step_forward(self, supply, physi_supply):
        self.step_parameter()

        self.recv_supply(supply, physi_supply)
        expected_delta_deploy = self.adjust_deploy()

        signal = self.send_signal()
        credit = self.gen_credit_score()

        self.step += 1

        return signal, credit, self.deploy_vm_num

 

    def recv_supply(self, supply, physi_supply):
        """
        Receive a supply quota.
        No return

        Input:
        - supply: an integer, denoting approved quota quantity, could be negative
        """
        assert type(supply) is int, "Supply TypeError"
        # update 1. supply history; 2. current quota; 3. quota history
        self.supply_hist.append(supply)
        self.cur_quota += supply
        # reset current quota physi
        # if self.cur_quota <= limit:
        self.cur_quota_physi = physi_supply
        # if self.cur_quota is way-too large, the transiton evaluation 
        # function will reset the self.cur_quota_physi to 
        # meaningfule value by self.update_physical_quota
        self.quota_hist.append(self.cur_quota)


    # def update_physical_quota(self, x):
    #     self.cur_quota_physi = x


    def gen_credit_score(self):
        """
        Generate credit score by some methods 
        Exponential decay
        """
        def exp_decay_cs():
            gamma_0, delta = 1, 2e-4
            K = self.credit_hist_len
            prev_cr = self.credit_hist[-1]
            if self.cur_quota == 0:
                dr = 0.0
            else:
                dr = gamma_0 * self.deploy_vm_num / self.cur_quota
            a = prev_cr * (K - 1)
            b = delta * self.deploy_ratio_sum
            new_credit = (dr + a - b) / K
            return new_credit

        new_credit_score = exp_decay_cs()

        # update 1. credit history; 2. current credit score
        self.credit_score = new_credit_score
        self.credit_hist.append(new_credit_score)
        self.credit_hist_len += 1
        return new_credit_score

    def step_parameter(self):
        """
        Update parameter 
        x1 - p1, p2, p3
           - p1, config['request'][0]
           - p3, config['recycle'][0]
        x1 = 0 - labda_11, config['request'][1]
               - r_11, config['request'][2]
        x1 = 2 - labda_12, config['recycle'][1]
               - r_12, config['recycle'][2]
        """
        # lb, ub = 0.2, 0.8
        # c = 0.1
        window_size = 10
        lr = 1
        alpha_bnd = {
            'request' : [0.6 * self.config['request'][1], 1.4 * self.config['request'][1]],
            'recycle' : [0.6 * self.config['recycle'][1], 1.4 * self.config['recycle'][1]],
        }
        scale_bnd= {
            'request' : [0.6 * self.config['request'][2], 1.4 * self.config['request'][2]],
            'recycle' : [0.6 * self.config['recycle'][2], 1.4 * self.config['recycle'][2]],
        }
       

        def step_x1():
            a = np.array(self.signal_hist[-window_size-1:-1])
            b = np.array(self.supply_hist[-window_size:]) 
            supply_window = b / (a + 1e-4)
            mu = supply_window.mean()
            sigma = supply_window.std()
            
            # p1 = (lb - 1) * mu + (ub - lb) + c * sigma
            # p3 =  (1 - lb) * mu + lb + c * sigma
            # p1 = max(min(p1, ub), lb)
            # p3 = max(min(p3, ub), lb)
            # p2 = 1 - p1 - p3
            a = [-.2, .0, .2]
            b = [-.2, .5, -.3]
            c = [.5, .2, .3]
            p1 = a[0] * mu + b[0] * sigma + c[0]
            p2 = a[1] * mu + b[1] * sigma + c[1]
            p3 = a[2] * mu + b[2] * sigma + c[2]
            assert abs(1 - (p1 + p2 + p3)) < 1e-7, "Consistency Error"

            self.config['request'][0] = p1
            self.config['recycle'][0] = p3

        def gamma_param(deploy_hist, key):
            prev_mu, prev_var = self.prev_mu[key], self.prev_var[key]
            if len(deploy_hist) == 0:
                # no positive or negative history
                mu1, var1 = prev_mu, prev_var
            else:
                deploy_window = np.array(deploy_hist[-window_size:])
                mu1 = deploy_window.mean()
                var1 = deploy_window.var() 

            Q = np.array(self.quota_hist)
            D = np.array(self.deploy_hist)
            t = Q - D
            # assert all(t >= 0), "Difference between Q and D may be wrong."
            mu2, var2 = t.mean(), t.var()

            mu = (mu1 + mu2) / 2
            var = (var1 + var2) / 2
            if mu < prev_mu:
                sign = -1
            else:
                sign = 1
            alpha = self.default_config[key][1]
            scale = self.default_config[key][2]
            if abs(var) > 1e-4:
                del_alpha = mu * mu / var
                del_scale = mu / var
                alpha += sign * lr * del_alpha
                scale += sign * lr * del_scale
                alpha = min(max(alpha, alpha_bnd[key][0]), alpha_bnd[key][1])
                scale = min(max(scale, scale_bnd[key][0]), scale_bnd[key][1])
                # beta = var / mu
                # scale = 1. / beta

            self.prev_mu[key], self.prev_var[key] = mu, var
            return alpha, scale

        def step_x2():
            t = np.array(self.delta_deploy_hist)
            posi_dh = t[t>0]
            neg_dh = t[t<0]

            self.config['request'][1:] = gamma_param(posi_dh, 'request')
            self.config['recycle'][1:] = gamma_param(-neg_dh, 'recycle')  # positate all negative numbers
        
        if self.is_dynamic:
            if self.step >= window_size:
                step_x1()
                step_x2()
        self.param_hist.append(
            self.config['request'] + self.config['recycle']
        )

    def get_dist(self, keyword):
        """
        Input:
        keyword - request, deploy
        Get distribution from config.
        """
        if keyword == 'request':
            p0 = self.config['request'][0]
            p2 = self.config['recycle'][0]
        elif keyword == 'deploy':
            p0 = self.config['deploy'][0]
            p2 = self.config['release'][0]
        else:
            raise NotImplementedError
        p1 = 1 - p0 - p2
        assert all([p0 >= 0, p1 >= 0, p2 >= 0]), "Distribution Error"
        dist_F = [p0, p0+p1, p0+p1+p2]

        assert abs(dist_F[-1] - 1.0) < 1e-7, "Config: deploy_type_dist SumError" 
        return dist_F

    def get_dr(self):
        """Deployment Ratio"""
        q1 = self.deploy_hist
        q2 = self.quota_hist
        q =  np.array(q1) / (np.array(q2) + 1e-7)

        # df = pd.DataFrame({'dep':q1, 'quo':q2, 'ratio':q})
        # out = df[df['ratio'] > 1.0]
        # if len(out) > 0:
        #     print(out)
        #     # raise ValueError
        #     print("WARNING: Deployment ratio over 1")
        return q

    def get_sr(self):
        """Satisfaction Ratio"""
        p1 = self.supply_hist
        p2 = self.signal_hist[:-1]
        p = np.array(p1) / (np.array(p2) + 1e-7)
        return p






