from rcrl.envs.common.subscriber import Subscriber
import numpy as np


class DoubleSubscriber(Subscriber):

    def adjust_deploy(self):
        """
        Adjust deployment number of virtual machine.
        """
        dist_F = self.get_dist(keyword='deploy')

        signal_type = np.random.random()
        # n means delta deploy
        # upper bound: current quota - current deploy
        # lower bound: -current deploy
        if signal_type <= dist_F[0]:
            shape, scale = self.config['deploy'][1:]
            n = int(np.random.gamma(shape, scale, size=1))
            delta_deploy = n
        elif signal_type <= dist_F[1]:
            delta_deploy = 0
        else:
            shape, scale = self.config['release'][1:]
            n = int(np.random.gamma(shape, scale, size=1))
            delta_deploy = -n

        # return intention to affect request signal
        expected_delta_deploy = delta_deploy

        delta_deploy_ub = self.cur_quota - self.deploy_vm_num
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

   





