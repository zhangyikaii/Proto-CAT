from models.utils import get_command_line_parser, gpu_state, pprint, alchemy_log

import os
import itertools
import time
from datetime import datetime
from copy import deepcopy

from multiprocessing import Process

gpus = '0,1'
params = {
    'temperature': [64, 512],
    'weijing_lambda_init': [0.1, 1],
    'lr': [0.0001, 0.00001, 0.000001],
}
space_hold = 8000 # 每个实验预留的GPU空间, 单GPU.
space_for_shixiong = 0 # 留给师兄的空间.
polling_interval = {'success': 120, 'fail': 60} # 单位: 秒.
manual_exec_dicts = [
    {
        'lr': 0.0003,
        'lr_scheduler': ['multistep'],
        'step_size': '30,50,100,160'
    }
    ]

def exec_args(args, exec_dict, is_first):
    store_true_params = ['do_train', 'do_test', 'epoch_verbose', 'verbose', 'weijing_is_learn_center', 'is_alchemy', 'weijing_is_hc']
    list_params = ['lr_scheduler']

    gpu_available = ''
    while len(gpu_available) == 0:
        if not is_first:
            time.sleep(polling_interval['fail'])
        gpu_space_available = gpu_state(gpus, get_return=True)
        for gpu_id, space in gpu_space_available.items():
            if space - space_for_shixiong >= space_hold:
                gpu_available = gpu_id
                break

    args.gpu = gpu_available
    args.time_str = datetime.now().strftime('%m%d-%H-%M-%S-%f')[:-3]

    command = 'nohup python ../main.py ' + ' '.join([f'--{param_name} {param_value}' for param_name, param_value in vars(args).items() if param_name not in store_true_params and param_name not in list_params]) + ' '
    command += ' '.join([f'--{param_name} ' + ' '.join(vars(args)[param_name]) for param_name in list_params]) + ' '
    command += ' '.join([f'--{param_name}' for param_name in store_true_params if vars(args)[param_name] is True])
    command += f' >> ./z_nohup_logs/{args.time_str}.log 2>&1 &'
    """
    nohup python ../main.py --data_path /data/zhangyk/data --model_class WeijingNet --backbone_class Res12 --distance l2 --dataset MiniImageNet --train_way 5 --val_way 5 --test_way 5 --train_shot 1 --val_shot 1 --test_shot 1 --train_query 15 --val_query 15 --test_query 15 --temperature 64.0 --lr 0.0001 --lr_mul 2 --step_size 50 --gamma 0.8 --init_weights /home/zhangyk/pre_trained_weights --weijing_mode 1 --gpu 0 --time_str 0802-16-30-26-134 --lr_scheduler step --do_train --do_test --epoch_verbose --verbose >> ./z_nohup_logs/0802-16-30-26-134.log 2>&1 &
    """
    log_str = f'{args.time_str} | {args.weijing_mode}: {exec_dict} ({gpu_available}).'
    print(log_str)
    print(command)
    alchemy_log(log_str)
    # alchemy_log(command + '\n')

    os.system(command)
    # trainer[args.time_str] = Trainer(args)
    # assert 0 # TODO: 这里是子进程还是命令行执行还有待商榷.
    # p = Process(target=trainer[args.time_str].fit, args=(True,))
    # p.start()

    time.sleep(polling_interval['success'])


def main():
    alchemy_log(f'\n{datetime.now().strftime("%m-%d %H:%M")}:')
    parser = get_command_line_parser(is_alchemy=True)
    args_init = parser.parse_args()

    params_keys = list(params.keys())

    is_first = True
    for params_tuple in itertools.product(*[i for i in params.values()]):
        cur_args = deepcopy(args_init)
        cur_exec_dict = {params_keys[i_set]: params_tuple[i_set] for i_set in range(len(params_keys))} # 构造key对应全组合的value的dict.
        for i_args, i_val in cur_exec_dict.items():
            exec(f'cur_args.{i_args} = {i_val}')

        exec_args(cur_args, cur_exec_dict, is_first)
        if is_first:
            is_first = False
    # TODO: 加一个关于 manual_exec_dicts 的循环, 先测上面可不可以.

if __name__ == '__main__':
    main()