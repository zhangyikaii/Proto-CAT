from models.utils import get_command_line_parser, gpu_state, pprint, alchemy_log

import os
import time
from datetime import datetime
from copy import deepcopy

from multiprocessing import Process

gpus = '0,1'

space_hold = 20000 # 每个实验预留的GPU空间, 单GPU.
space_for_shixiong = 0 # 留给师兄的空间.
polling_interval = {'success': 120, 'fail': 60} # 单位: 秒.
manual_exec_dicts = [
    {
        'do_train': False,
        'model_class': 'MultiModalProtoNet',
        'backend_type': ['LSTM', 'GRU'],
        'mm_loss_video': False,
        'mm_loss_audio': True
    },
    {
        'do_train': False,
        'model_class': 'MultiModalProtoNet',
        'backend_type': ['LSTM', 'LSTM'],
        'mm_loss_video': False,
        'mm_loss_audio': True
    },
    {
        'do_train': False,
        'model_class': 'MultiModalProtoNet',
        'backend_type': ['LSTM', 'MSTCN'],
        'mm_loss_video': False,
        'mm_loss_audio': True
    }
    ]

def exec_args(args, exec_dict, is_first):
    store_true_params = ['do_train', 'do_test', 'epoch_verbose', 'verbose', 'is_alchemy', 'grad_scaler', 'gfsl_train', 'gfsl_test', 'acc_per_class', 'mm_loss_video', 'mm_loss_audio']
    list_params = ['lr_scheduler', 'mm_list', 'backend_type']

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

    command = 'nohup python ../main.py ' + ' '.join([f'--{param_name} {param_value}' for param_name, param_value in vars(args).items() if (param_name not in store_true_params) and (param_name not in list_params) and (not (isinstance(param_value, str) and len(param_value) == 0))]) + ' '
    command += ' '.join([f'--{param_name} ' + ' '.join(vars(args)[param_name]) for param_name in list_params]) + ' '
    command += ' '.join([f'--{param_name}' for param_name in store_true_params if vars(args)[param_name] is True])
    command += f' >> ./z_nohup_logs/{args.time_str}.log 2>&1 &'

    """
    0909-08-00-36-745 | MultiModalProtoNet LSTM: {'model_class': 'MultiModalProtoNet', 'backend_type': 'LSTM', 'mm_loss_video': False, 'mm_loss_audio': False} (0).
    nohup python ../main.py --data_path /data/zhangyk/data --model_class MultiModalProtoNet --distance l2 --dataset LRW --max_epoch 50 --train_way 5 --val_way 5 --test_way 5 --train_shot 1 --val_shot 1 --test_shot 1 --train_query 15 --val_query 15 --test_query 15 --temperature 64.0 --lr 0.0001 --lr_mul 10 --step_size 20 --gamma 0.7 --init_weights /home/zhangyk/pre_trained_weights --cosine_annealing_lr_eta_min 5e-06 --inlayer_resnet_type resnet18 --backend_type LSTM --gpu 0 --time_str 0909-08-00-36-745 --lr_scheduler cosine cosine --mm_list video audio --do_train --do_test --epoch_verbose --verbose --is_alchemy --grad_scaler --gfsl_test --acc_per_class >> ./z_nohup_logs/0909-08-00-36-745.log 2>&1 &
    """
    log_str = f'{args.time_str} | {args.model_class} {args.backend_type}: {exec_dict} ({gpu_available}).'
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
    parser = get_command_line_parser(is_alchemy_sequential=True)
    args_init = parser.parse_args()

    is_first = True
    for params_tuple in manual_exec_dicts:
        cur_args = deepcopy(args_init)
        for i_args, i_val in params_tuple.items():
            if isinstance(i_val, str):
                exec(f'cur_args.{i_args} = i_val')
            else:
                exec(f'cur_args.{i_args} = {i_val}')

        exec_args(cur_args, params_tuple, is_first)
        if is_first:
            is_first = False
    # TODO: 加一个关于 manual_exec_dicts 的循环, 先测上面可不可以.

if __name__ == '__main__':
    main()