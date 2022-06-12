from rcrl.envs.common.subscriber import Subscriber
import numpy as np

class SingleSubscriber(Subscriber):
    ratio_of_deploy = 0.7

    def send_signal(self):
        """
        Sample the new request signal from a parametrized gamma distribution.
        Return an integer signal.

        Output:
        - signal number: 
            - if s > 0, ask for quota; 
            - s = 0, no action; 
            - s < 0, give back quota;

        When deploy is correlated with signal, signal could not depend on 
        deploy anymore.
        """

        def gen_request():
            shape, scale = self.config['request'][1:]
            n = int(np.random.gamma(shape, scale, size=1))  # return a np.ndarray
            return n

        def gen_recycle():
            shape, scale = self.config['recycle'][1:]
            n = int(np.random.gamma(shape, scale, size=1))
            n = max(-n, -self.cur_quota)
            return n 
        dist_F = self.get_dist(keyword='request')
        signal_type = np.random.random()
        if signal_type <= dist_F[0]:
            new_signal = gen_request()
        elif signal_type <= dist_F[1]:
            new_signal = 0
        else:
            new_signal = gen_recycle()
            # delta deploy < 0
            # if self.behave == "good":
                # new_signal = gen_recycle()
            # elif self.behave == "bad":
                # new_signal = gen_request()
            # else:
                # new_signal = 0
        # save value for correlating expected deployment
        self.cur_signal = new_signal
        # update signal history
        self.signal_hist.append(new_signal)
        return new_signal

    def adjust_deploy(self):
        """
        Adjust deployment number of virtual machine.
        """
        delta_deploy = int(self.ratio_of_deploy * self.cur_signal)
        # return intention to affect request signal
        expected_delta_deploy = delta_deploy

        # cur_quota_physi <= cur_quota
        delta_deploy_ub = self.cur_quota_physi - self.deploy_vm_num
        if delta_deploy > delta_deploy_ub:
            # bad allocation plan
            self.confined_hist.append(True)
        else:
            self.confined_hist.append(False)

        # intrinsic range
        delta_deploy = min(delta_deploy, delta_deploy_ub)
        delta_deploy = max(delta_deploy, -self.deploy_vm_num)

        self.delta_deploy_hist.append(delta_deploy)
        self.deploy_vm_num += delta_deploy 
        self.deploy_hist.append(self.deploy_vm_num)
        return expected_delta_deploy

class GoodSubscriber(SingleSubscriber):
    ratio_of_deploy = 0.7

class BadSubscriber(SingleSubscriber):
    ratio_of_deploy = 0.3


class GreedySubscriber(SingleSubscriber):

    def step_forward(self, supply, signal):
        self.step_parameter()

        self.recv_supply(supply, supply)
        expected_delta_deploy = self.adjust_deploy()

        # receive a pre-generated signal from outside
        credit = self.gen_credit_score()

        self.step += 1

        return signal, credit, self.deploy_vm_num