from configparser import ConfigParser

config_object = ConfigParser()

# torch_seed, cuda_seed, np_seed
config_object["SEEDS_SEARCH_RANGE_INFO"] = {
    "f_torch_seed": "1",
    "t_torch_seed": "100",
    "f_cuda_seed": "1",
    "t_cuda_seed": "100",
    "f_np_seed": "1",
    "t_np_seed": "100",
    "step": "1"
}

with open('config.ini', 'w') as conf:
    config_object.write(conf)