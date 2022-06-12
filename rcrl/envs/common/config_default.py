# =============================
#  Subscriber Related Config
# =============================

# initial quota >= 10 * sum of mean
# initial quota >= 10 * 10*(100 + 50 + 100 + 50)

abundant_init_ava_quota = 20000
deficient_init_ava_quota = 2000
# deficient_init_ava_quota = 100

init_credit = [0.3, 0.8, 0.5, 0.35]
init_cur_quota = [300, 300, 300, 300]
init_deploy = [200, 100, 150, 180]

name_to_config_exp_small = {
    'Alice' : {
        'init_deploy' : init_deploy[0],
        'credit' : init_credit[0],
        'cur_quota' : init_cur_quota[0], 
        "config" : {
            "request" : [0.5, 50, 2],  # mean = 50 * 2 = 100
            "recycle" : [0.4, 50, 2],
            "deploy"  : [0.5, 40, 2],
            "release" : [0.4, 40, 2],
            "behave" : "good",
        }
    }, 
    'Bob' : {
        'init_deploy' : init_deploy[1],
        'credit' : init_credit[1],
        'cur_quota' : init_cur_quota[1], 
        "config" : {
            "request" : [0.5, 25, 2],  # mean = 25 * 2 = 50
            "recycle" : [0.4, 25, 2],
            "deploy"  : [0.5, 20, 2],
            "release" : [0.4, 20, 2],
            "behave" : "bad",
        }
    },  
    'Sam' : {
        'init_deploy' : init_deploy[2],
        'credit' : init_credit[2],
        'cur_quota' : init_cur_quota[2], 
        "config" : {
            "request" : [0.3, 50, 2],
            "recycle" : [0.2, 50, 2],
            "deploy"  : [0.3, 40, 2],
            "release" : [0.2, 40, 2],
            "behave" : "good",
        }
    },  
    'David' : {
        'init_deploy' : init_deploy[3],
        'credit' : init_credit[3],
        'cur_quota' : init_cur_quota[3], 
        "config" : {
            "request" : [0.3, 25, 2],
            "recycle" : [0.2, 25, 2],
            "deploy"  : [0.3, 20, 2],
            "release" : [0.2, 20, 2],
            "behave" : "bad",
        }
    }
}

name_to_config_exp = name_to_config_exp_small

def get_larger_name_to_config(users_per_group=10):
    name_to_group = dict()
    for name, config in name_to_config_exp_small.items():
        for suffix in range(users_per_group):
            new_name = name + "_" + str(suffix)
            name_to_group[new_name] = config # absolute initial state
    return name_to_group


# switch to open larger user config
# deficient_init_ava_quota = 2000
# name_to_config_exp = get_larger_name_to_config()
# init_cur_quota = [config['cur_quota'] for _, config in name_to_config_exp.items()]
# init_deploy = [config['init_deploy'] for _, config in name_to_config_exp.items()]









