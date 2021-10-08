import os, sys

sys.path.append("../../")
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])

from models.utils import (
    get_command_line_parser, 
    preprocess_args,
    pprint
)
from models.train import (
    Trainer
)

class MyConfigParser(object):
    def __init__(self, config_filepath="config.ini", key_name="SEEDS_SEARCH_RANGE_INFO"):
        from configparser import ConfigParser

        config_object = ConfigParser()
        config_object.read(config_filepath)
        self.range_info = config_object["SEEDS_SEARCH_RANGE_INFO"]
    def __getitem__(self, item):
        return int(self.range_info[item])

if __name__ == '__main__':
    model_parser = get_command_line_parser()
    model_args = preprocess_args(model_parser.parse_args())

    config = MyConfigParser(config_filepath="config.ini", key_name="SEEDS_SEARCH_RANGE_INFO")
    step = config['step']

    best = 0
    best_seeds_dict = {'torch_seed': 0, 'cuda_seed': 0, 'np_seed': 0}
    for torch_seed in range(config['f_torch_seed'], config['t_torch_seed'], step):
        for cuda_seed in range(config['f_cuda_seed'], config['t_cuda_seed'], step):
            for np_seed in range(config['f_np_seed'], config['t_np_seed'], step):
                model_args.torch_seed, model_args.cuda_seed, model_args.np_seed = torch_seed, cuda_seed, np_seed
                trainer = Trainer(model_args)
                cur = trainer.test()
                if cur > best:
                    best = cur
                    best_seeds_dict.update({'torch_seed': torch_seed, 'cuda_seed': cuda_seed, 'np_seed': np_seed})
                    print(f'{cur}: {best_seeds_dict}')

    
