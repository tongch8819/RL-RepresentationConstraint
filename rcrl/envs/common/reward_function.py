import numpy as np
import pandas as pd

"""
[0, 10] is a decent range for every reward signal at timestep t.
sign has been set within the private functions.
"""
overflow_coef = 1e-2
hinderance_coef = 1e-2
deploy_coef = 5e-3
oaf_coef = 1e-1

def naive_reward(o, a):
    ao = _allocation_overflow(o, a)
    x1 = _hinderance(o, a)
    x2 = _deployment_number(o, a)
    r_comps = {
        'Overflow' : ao,
        'Hinderance' : x1, 
        'DeploymentNum' : x2,
    }
    coeffs = np.array([overflow_coef, hinderance_coef, deploy_coef])
    rews = np.array([ao, x1, x2])
    return np.sum(coeffs * rews), r_comps

def naive_reward_oaf(o, a):
    rew, r_comps = naive_reward(o, a)
    oaf_left, oaf_right = _oaf(o, a)
    r_comps.update({
        'AllocationFairnessCredit' : oaf_left,
        'AllocationFairnessDep' : oaf_right,
    })
    coeffs = np.array([oaf_coef] * 2)
    sub_rews = np.array([oaf_left, oaf_right])
    rew += np.sum(coeffs * sub_rews)
    return rew, r_comps

def primary_reward(o, a):
    ao = _allocation_overflow(o, a)
    r_comps = {
        'Overflow' : ao,
    }
    return overflow_coef * ao, r_comps 

def primary_reward_oaf(o, a):
    rew, r_comps = primary_reward(o, a)
    oaf_left, oaf_right = _oaf(o, a)
    r_comps.update({
        'AllocationFairnessCredit' : oaf_left,
        'AllocationFairnessDep' : oaf_right,
    })
    coeffs = np.array([oaf_coef] * 2)
    sub_rews = np.array([oaf_left, oaf_right])
    rew += np.sum(coeffs * sub_rews)
    return rew, r_comps

def secondary_reward(o, a):
    x1 = _hinderance(o, a)
    x2 = _deployment_number(o, a)
    r_comps = {
        'Hinderance' : x1, 
        'DeploymentNum' : x2,
    }
    return hinderance_coef * x1 + deploy_coef * x2, r_comps


def _hinderance(o2, a):
    hinder_cnt = (o2.allocated_vec == o2.deploy_vec).sum()
    return -hinder_cnt

def _deployment_number(o, a):
    res = o.deploy_vec[0]
    return res

def _allocation_overflow(state, action):
    """
    range: [-500, 0] 
    """
    is_overflow = False
    req = state.request_vec
    if hasattr(state, 'satisfaction_ratio'):
        sup = state.satisfaction_ratio
    elif hasattr(action, 'supply_ratio_vec'):
        sup = action.supply_ratio_vec
    else:
        raise ValueError("Satisfaction Ratio Unknown")
    # adjust supply vector to avoid impossible case: 
    # - sum of supply is larger than available quota
    h = state.available_quota[0]

    v = np.floor(req * sup)
    p = - sum(np.clip(v, a_max=0, a_min=None))
    q = sum(np.clip(v, a_max=None, a_min=0))
    assert q - p == sum(v), f"Consistency broken,\n{q}\n{p}\n{v}"
    # loss
    L = max(0, q - h - p)
    return -L

def _oaf(state, action):
    if hasattr(state, 'satisfaction_ratio'):
        sa = state.satisfaction_ratio
    elif hasattr(action, 'supply_ratio_vec'):
        sa = action.supply_ratio_vec
    else:
        raise ValueError("Satisfaction Ratio Unknown")
    cr = state.credit_vec
    dr = state.deploy_vec / (state.allocated_vec + 1e-4) + 1e-4
    # data = [sa, cr, dr]
    data = dict(sr=sa, cre=cr, dep=dr)
    core_df = pd.DataFrame(data).rank(
            method='max', na_option='bottom', ascending=True)
    a = (core_df['sr'] - core_df['cre']).abs().sum()
    b = (core_df['sr'] - core_df['dep']).abs().sum()
    return -a, -b



