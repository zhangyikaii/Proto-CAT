import os, shutil
import os.path as osp
import time
import torch
from torch.utils.data import DataLoader

from models.metrics import ROOT_PATH
import models.metrics as mtrcs

import numpy as np
import random
import copy

import pickle
import yaml

from scipy import sparse

from sklearn.neighbors import NearestNeighbors

from collections import OrderedDict


class CheckModelChanged():
    def __init__(self):
        self.m = None
    def set_model(self, model):
        assert self.m is None
        self.m = copy.deepcopy(model).to(torch.device('cuda'))

    def check_model(self, model):
        assert self.m is not None
        for (k1, v1), (k2, v2) in zip(self.m.state_dict().items(), model.state_dict().items()):
            if k1 != k2 or not torch.equal(v1, v2):
                print(f'The model has changed, please check key1: {k1} and key2: {k2}:')
                print(v1)
                print()
                print(v2)
                return False
        return True

class SetWithIndex():
    def __init__(self):
        self.s = {}
        self.cur_idx = 0
    def append(self, k):
        if k not in self.s.keys():
            self.s[k] = self.cur_idx
            self.cur_idx += 1
    def __getitem__(self, idx):
        return self.s[idx]
    def __len__(self):
        return self.cur_idx

# check_model_changed_handle = CheckModelChanged()

def set_seeds(torch_seed, cuda_seed, np_seed, random_seed):
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(cuda_seed)
    np.random.seed(np_seed)
    random.seed(random_seed)

def is_zero(x):
    return torch.abs(x) < 1e-6

def nan_assert(x):
    assert torch.any(torch.isnan(x)) == False

def mkdir(dirs):
    """Create a directory, ignoring exceptions

    # Arguments:
        dir: Path of directory to create
    """
    if not os.path.exists(dirs):
        os.makedirs(dirs)

def rmdir(dirs):
    """Recursively remove a directory and contents, ignoring exceptions

   # Arguments:
       dir: Path of directory to recursively remove
   """
    if os.path.exists(dirs):
        shutil.rmtree(dirs)


def get_label2id_global(dataset):
    with open(osp.join(ROOT_PATH, f'data/{dataset}/split/train.csv'), 'r') as f:
        label_meta_train = list(set([x.strip().split(',')[1] for x in f.readlines()][1:]))
    label2id = {k: i for i, k in enumerate(sorted(label_meta_train))}
    base_num = len(label2id)

    label_meta_test = []
    with open(osp.join(ROOT_PATH, f'data/{dataset}/split/val.csv'), 'r') as f:
        label_meta_test = list(set([x.strip().split(',')[1] for x in f.readlines()][1:]))
    with open(osp.join(ROOT_PATH, f'data/{dataset}/split/test.csv'), 'r') as f:
        label_meta_test.extend(list(set([x.strip().split(',')[1] for x in f.readlines()][1:])))
    label2id.update({k: i + base_num for i, k in enumerate(sorted(label_meta_test))})
    return label2id


def parse_dataloader_split_csv(data_root_path, csv_path):
    assert mtrcs.LABEL2ID is not None
    lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

    data = []
    label = []

    for l in lines:
        name, wnid = l.split(',')
        path = osp.join(data_root_path, name)

        data.append(path)
        label.append(mtrcs.LABEL2ID[wnid])

    return data, label

def get_mm_incomplete_id2label(label_incomplete_path, label_id_list=None):
    assert mtrcs.LABEL2ID is not None
    if osp.isfile(label_incomplete_path):
        with open(label_incomplete_path, 'r') as f:
            mm_incomplete_list = [(line.strip().split(',')[0], line.strip().split(',')[1]) for line in f]
    else:
        mm_incomplete_list = []

    if label_id_list is None:
        return dict(sorted({mtrcs.LABEL2ID[i[0]]: i for i in mm_incomplete_list}.items()))
    else:
        return dict(sorted({mtrcs.LABEL2ID[i[0]]: i for i in mm_incomplete_list if mtrcs.LABEL2ID[i[0]] in label_id_list}.items()))

def get_command_line_parser(is_alchemy=False, is_alchemy_sequential=False):
    """解析命令行参数.

    # Arguments
        None
    # Return
        argparse.ArgumentParser(), 还需要 .parse_args() 转.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train', action='store_true', default=False)
    parser.add_argument('--do_test', action='store_true', default=False)
    parser.add_argument('--data_path', type=str, default='/data/zhangyk/data')
    parser.add_argument('--model_class', type=str, default='ProtoNet',
                        choices=['MultiModalMAML', 'ProtoCAT_Plus', 'ProtoCAT', 'RGMultiModalProtoNetPlusCRS', 'RGMultiModalProtoNetSPPATTN', 'RGMultiModalProtoNetPlusWOATTN', 'RGMultiModalProtoNetPlus', 'CASTLE', 'RGTIMAPlus', 'MultiModalMatchingNet', 'RGTIMA', 'RGMultiModalProtoNet', 'WeijingNet', 'EfficientNet', 'LaplacianProtoNet', 'Rouge', 'Linear', 'MultiModalTIMAPlus', 'MultiModalTIMA', 'MultiModalProtoNet', 'RTPretrainClassifier', 'LinearClassifier', 'CRGPretrainClassifier', 'ProtoNetPretrainClassifier', 'MetaOptNet', 'MAML', 'MatchingNet', 'ProtoNet', 'BILSTM', 'DeepSet', 'GCN', 'RelationNet', 'FEAT', 'FEATSTAR', 'SemiFEAT', 'SemiProtoFEAT'])
    parser.add_argument('--distance', default='l2')
    parser.add_argument('--dataset', type=str, default='MiniImageNet',
                        choices=['MiniImageNet', 'TieredImageNet', 'CUB', 'ESC50', 'LRW', 'LRW1000'])
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--train_way', type=int, default=5)
    parser.add_argument('--val_way', type=int, default=5)
    parser.add_argument('--test_way', type=int, default=5)
    parser.add_argument('--train_shot', type=int, default=1)
    parser.add_argument('--val_shot', type=int, default=1)
    parser.add_argument('--test_shot', type=int, default=1)
    parser.add_argument('--train_query', type=int, default=15)
    parser.add_argument('--val_query', type=int, default=15)
    parser.add_argument('--test_query', type=int, default=15)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr_mul', type=float, default=10)
    parser.add_argument('--lr_scheduler', nargs='*', default=['step'])
    parser.add_argument('--step_size', type=str, default='20')
    parser.add_argument('--gamma', type=float, default=0.7)
    parser.add_argument('--epoch_verbose', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--is_alchemy', action='store_true', default=False)
    parser.add_argument('--cosine_annealing_lr_eta_min', type=float, default=5e-6)
    parser.add_argument('--grad_scaler', action='store_true', default=False)

    # MM:
    parser.add_argument('--mm_list', nargs='*', default=[])
    parser.add_argument('--inlayer_resnet_type', type=str, default='resnet18')
    parser.add_argument('--backend_type', nargs='*', default=[])
    parser.add_argument('--mm_incomplete_type_video', type=str, default='', choices=['noise', 'gaussian_blur', 'pepper_noise'])
    parser.add_argument('--mm_incomplete_type_audio', type=str, default='', choices=['background_noise', 'gaussian_noise'])
    parser.add_argument('--mm_loss_video', action='store_true', default=False)
    parser.add_argument('--mm_loss_audio', action='store_true', default=False)

    parser.add_argument('--gfsl_test', action='store_true', default=False)
    parser.add_argument('--gfsl_train', action='store_true', default=False)
    parser.add_argument('--acc_per_class', action='store_true', default=False)


    if is_alchemy_sequential:
        return parser

    parser.add_argument('--init_weights', type=str, default=None) # 多模态情况下为路径, 单模态为模型文件.
    parser.add_argument('--backbone_class', type=str, default='ConvNet',
                        choices=['ConvNet', 'Conv4', 'Conv4NP', 'Res12', 'Res18', 'WRN', 'Linear', 'Conv3dResNet', 'Conv1dResNet', 'MLGCN', 'MetaLearner'])

    parser.add_argument('--weijing_mode', type=str, default='')
    parser.add_argument('--weijing_is_learn_center', action='store_true', default=False)
    parser.add_argument('--weijing_lambda_init', type=float, default=0.1)
    parser.add_argument('--weijing_is_hc', action='store_true', default=False)
    

    if is_alchemy:
        return parser

    parser.add_argument('--gpu_space_require', type=int, default=24000)

    parser.add_argument('--train_monitor', type=str, default='categorical accuracy')
    parser.add_argument('--num_tasks', type=int, default=4)
    parser.add_argument('--episodes_per_train_epoch', type=int, default=200)
    parser.add_argument('--episodes_per_val_epoch', type=int, default=200)
    parser.add_argument('--episodes_per_test_epoch', type=int, default=10000)
    parser.add_argument('--drop_lr_every', type=int, default=40)
    parser.add_argument('--logger_filename', type=str, default='/z_logs')
    # parser.add_argument('--balance', type=float, default=1)
    parser.add_argument('--paradigm', nargs='+', default=['few-shot', 'few-shot', 'few-shot']) # 决定各种Dataloader等要素的类型.

    parser.add_argument('--temperature2', type=float, default=1)  # the temperature in the  
    parser.add_argument('--loss_fn', type=str, default='F-cross_entropy',
                        choices=['F-cross_entropy', 'nn-cross_entropy', 'nn-nll', 'nn-mse'])
    parser.add_argument('--do_prefetch', action='store_true', default=False)

    # optimization parameters
    parser.add_argument('--orig_imsize', type=int, default=-1) # -1 for no cache, and -2 for no resize, only for MiniImageNet and CUB
    parser.add_argument('--fix_BN', action='store_true', default=False) # means we do not update the running mean/var in BN, not to freeze BN
    parser.add_argument('--augment', action='store_true', default=False)
    parser.add_argument('--without_enlarge', action='store_true', default=False)

    parser.add_argument('--gpu', default='0')
    parser.add_argument('--meta_batch_size', type=int, default=1)
    parser.add_argument('--time_str', type=str, default='')

    # usually untouched parameters
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--val_interval', type=int, default=1)
    parser.add_argument('--test_interval', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')

    parser.add_argument('--model_save_path', type=str, default='/data/zhangyk/models')
    parser.add_argument('--test_model_filepath', type=str, default=None)

    parser.add_argument('--torch_seed', type=int, default=929)
    parser.add_argument('--cuda_seed', type=int, default=929)
    parser.add_argument('--np_seed', type=int, default=929)
    parser.add_argument('--random_seed', type=int, default=929)

    # MatchingNet:
    parser.add_argument('--matching_fce', action='store_true', default=False)
    parser.add_argument('--matching_normalized', action='store_true', default=False)
    parser.add_argument('--matching_lstm_layers', type=int, default=1)
    parser.add_argument('--matching_unrolling_steps', type=int, default=2)

    # RelationNet:
    parser.add_argument('--relation_hidden_dim', type=int, default=8)

    # MAML:
    parser.add_argument('--meta', action='store_true', default=False)
    parser.add_argument('--inner_train_steps', type=int, default=1)
    parser.add_argument('--inner_val_steps', type=int, default=3)
    parser.add_argument('--inner_iters', type=int, default=8)
    parser.add_argument('--gd_lr', type=float, default=0.005)
    parser.add_argument('--gd_weight_decay', type=float, default=0.00005)
    parser.add_argument('--gd_mom', type=float, default=0.9)

    # pretrain:
    parser.add_argument('--batch_size', type=int, default=128)
    # parser.add_argument('--schedule', type=int, nargs='+', default=[75, 150, 300],
    #     help='Decrease learning rate at these epochs (Use only during pre-training).')
    # parser.add_argument('--resume', action='store_true', default=False)

    # # RouGee
    # parser.add_argument('--rougee', action='store_true', default=False)
    # parser.add_argument('--rougee_num_nn', type=int, default=1)
    # parser.add_argument('--rougee_episodes_per_epoch', type=int, default=10000)
    # parser.add_argument('--rougee_norm_type_list', type=str,
    #                     default='un,l2n,cl2n,meta_test_zca_whitening,meta_test_pca_whitening,meta_test_pca_pure,meta_test_zca_corr_whitening,meta_test_pca_corr_whitening,meta_train_zca_whitening,meta_train_pca_whitening,meta_train_pca_pure,meta_train_zca_corr_whitening,meta_train_pca_corr_whitening')

    parser.add_argument('--z_comment', type=str, default='Here are some comments for the current training process.')

    # Rouge:
    parser.add_argument('--nearest_topk', type=int, default=3)
    parser.add_argument('--tr_pca_n_components', type=int, default=None)
    parser.add_argument('--tr_pca_ratio', type=float, default=0.95)
    parser.add_argument('--rougee_cache_path', type=str, default=None)
    parser.add_argument('--tensorboard_log_dir', type=str, default='/z_logs/ycy_tensorboard')
    parser.add_argument('--rougee_norm_type_list', nargs='+', default=['ori__ori__prt'])

    parser.add_argument('--laplacian_protonet_query_distance_temperature', type=float, default=4)
    parser.add_argument('--laplacian_protonet_delta', type=float, default=0.3)
    parser.add_argument('--laplacian_protonet_lambda', type=float, default=6)

    parser.add_argument('--metalearner_w_dim', type=int, default=64)
    parser.add_argument('--metalearner_update_step', type=int, default=4)
    parser.add_argument('--metalearner_update_lr', type=float, default=0.0001)

    parser.add_argument('--weijing_is_learn_lambda', action='store_true', default=False)
    parser.add_argument('--weijing_lambda_base', type=float, default=1.0)
    parser.add_argument('--weijing_lap_delta', type=float, default=64.0)
    parser.add_argument('--weijing_base_feature_update_tiktok', type=int, default=40)
    parser.add_argument('--weijing_multiview', action='store_true', default=False)
    parser.add_argument('--weijing_multiview_num', type=int, default=4)

    return parser

class Timer():
    def __init__(self, total):
        self.o = time.time()
        self.total = total

    def time2str(self, x):
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

    def measure(self, used=0):
        cur_used = time.time() - self.o
        if used == 0:
            return self.time2str(cur_used), self.time2str(cur_used * self.total)
        else:
            return self.time2str(cur_used), self.time2str(cur_used / used * (self.total - used))

class Averager():
    def __init__(self):
        self.n = 0
        self.v = 0
    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1
    def item(self):
        return self.v

class FastAverager():
    def __init__(self):
        self.n = 0
        self.v = 0
    def add(self, x):
        self.v += x
        self.n += 1
    def reset(self):
        self.__init__()
    def item(self):
        return self.v / self.n

class FastAveragerDict():
    def __init__(self):
        self.d = {}
    def add(self, x):
        for i in x.keys():
            if i not in self.d.keys():
                self.d[i] = [x[i]]
            else:
                self.d[i].append(x[i])
    def reset(self):
        self.__init__()
    def item(self):
        ret = {}
        for i, v in self.d.items():
            ret[i] = sum(v) / len(v)
        return ret
    # def get(self):
    #     return self.d

import pprint
_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)

def set_logger(filename, logger_name):
    mkdir(filename[:filename.rfind('/')])
    import logging
    logging.basicConfig(
        filename=filename,
        filemode='a',
        level=logging.DEBUG,
        format='%(asctime)s %(message)s',
        datefmt='%Y/%m/%d %I:%M:%S %p'
    )
    return logging.getLogger(logger_name)

def gpu_state(gpu_id, get_return=False):
    qargs = ['index', 'gpu_name', 'memory.used', 'memory.total']
    cmd = 'nvidia-smi --query-gpu={} --format=csv,noheader'.format(','.join(qargs))

    results = os.popen(cmd).readlines()
    gpu_id_list = gpu_id.split(",")
    gpu_space_available = {}
    for cur_state in results:
        cur_state = cur_state.strip().split(", ")
        for i in gpu_id_list:
            if i == cur_state[0]:
                if not get_return:
                    print(f'GPU {i} {cur_state[1]}: Memory-Usage {cur_state[2]} / {cur_state[3]}.')
                else:
                    gpu_space_available[i] = int("".join(list(filter(str.isdigit, cur_state[3])))) - int("".join(list(filter(str.isdigit, cur_state[2]))))
    if get_return:
        return gpu_space_available

def set_gpu(x, space_hold=20000):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    gpu_available = 0
    while gpu_available < space_hold:
        gpu_space_available = gpu_state(x, get_return=True)
        for gpu_id, space in gpu_space_available.items():
            gpu_available += space
        if gpu_available < space_hold:
            print(f'当前GPU(s)剩余: {gpu_available}.')
            gpu_available = 0
            time.sleep(1800) # 间隔30分钟.
    gpu_state(x)

def preprocess_args(args):
    """根据命令行参数附加处理.

    添加参数:
        TODO

    # Argument
        args: parser.parse_args()
    # Return
        处理后的args
    """
    # TODO: setup_dirs 应该在这里根据传入参数建.

    if len(args.lr_scheduler) == 1:
        args.lr_scheduler = args.lr_scheduler[0]
    # 添加由数据集决定的参数:
    if args.dataset == 'OmniglotDataset':
        args.num_input_channels = 1
    elif args.dataset == 'MiniImageNet':
        args.num_input_channels = 3

    if args.time_str == '':
        from datetime import datetime
        args.time_str = datetime.now().strftime('%m%d-%H-%M-%S-%f')[:-3]

    # args.mm_list = list(set(args.mm_train_list) | set(args.mm_test_list))
    # mm_train_list 和 mm_test_list 是控制是否multimodal的核心参数, 将决定unimodal参数.
    if len(args.mm_list) == 0:
        args.unimodal = True
        if args.backbone_class in ['Conv3dResNet', 'Conv1dResNet']:
            bkb2mdl_dict = {
                'Conv3dResNet': 'video',
                'Conv1dResNet': 'audio'
                }
            args.unimodal_class = bkb2mdl_dict[args.backbone_class]
        args.params_str = f'{args.time_str} {args.model_class} {args.dataset} {args.backbone_class}-backbone {args.distance} ' \
            f'{args.train_way}-{args.train_shot}-{args.train_query}_train-w-s-q {args.val_way}-{args.val_shot}-{args.val_query}_val-w-s-q {args.test_way}-{args.test_shot}-{args.test_query}_test-w-s-q'
    else:
        args.unimodal = False
        mdl2bkb_dict = {
            'video': 'Conv3dResNet',
            'audio': 'Conv1dResNet'
        }
        args.backbone_class = [mdl2bkb_dict[i] for i in args.mm_list]
        args.params_str = f'{args.time_str} {args.model_class} {args.dataset} {("_".join(args.backbone_class))}-backbone {args.distance} ' \
            f'{args.train_way}-{args.train_shot}-{args.train_query}_train-w-s-q {args.val_way}-{args.val_shot}-{args.val_query}_val-w-s-q {args.test_way}-{args.test_shot}-{args.test_query}_test-w-s-q'

    # test_model_filepath 优先级高于 model_save_path.
    if args.test_model_filepath is not None:
        assert not args.do_train
        args.model_filepath = args.test_model_filepath
    else:
        args.model_filepath = f'{args.model_save_path}/{args.model_class}/{args.params_str}.pth'
    # 在此之后 test_model_filepath 没有用了, 因为已经传递给model_filepath的.


    paradigm_tmp = []
    for i in args.paradigm:
        paradigm_tmp.append('1' if i == 'few-shot' else '0')
    args.paradigm = int(''.join(paradigm_tmp), 2)

    return args

def create_query_label(way: int, query: int) -> torch.Tensor:
    """Creates an shot-shot task label.

    Label has the structure:
        [0]*query + [1]*query + ... + [way-1]*query

    # Arguments
        way: Number of classes in the shot-shot classification task
        query: Number of query samples for each class in the shot-shot classification task

    # Returns
        y: Label vector for shot-shot task of shape [query * way, ]
    """

    # 返回从 0 ~ way - 1 (label), 每个元素有 query 个(query samples).
    return torch.arange(0, way, 1 / query).long() # 很精妙, 注意强转成long了.

def create_onehot_query_label(way: int, shot: int) -> torch.Tensor:
    return torch.zeros((way * shot, way)).scatter_(1, create_query_label(way, shot).unsqueeze(1), 1).long()

def to_one_hot(y, num_classes):
    return torch.zeros((len(y), num_classes)).scatter_(1, y.unsqueeze(1), 1)

def pretrain_prepare_batch(batch):
    x, y = batch
    return x.to(torch.device('cuda')), y.to(torch.device('cuda'))

def multimodal_pretrain_prepare_batch(batch):
    x, y = batch
    for k in x.keys():
        x[k] = x[k].to(torch.device('cuda'))
    return x, y.to(torch.device('cuda'))

def update_add_dicts(x, add_to_x):
    for i in add_to_x.keys():
        x[i] = add_to_x[i] if i not in x.keys() else x[i] + add_to_x[i]

def divide_dict(x, a, k):
    for i in k:
        x[i] /= a

def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data, 0.95 * standard error.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm

def torch_cov(x, cur_mean):
    c = x - cur_mean
    return (1.0 / (x.shape[0] - 1)) * (torch.transpose(c, -2, -1) @ c)

def torch_z_score(x):
    # Test it:
    # from scipy import stats
    # print(stats.zscore(x.cpu().detach().numpy()))

    cur_std = torch.std(x, dim=0, unbiased=False, keepdim=True)

    # cur_std[cur_std == 0] = 1e-5

    return (x - torch.mean(x, dim=0, keepdim=True)) / (cur_std + 1e-5)

def torch_cl2n(s, q, train_mean):
    support = s - train_mean
    query = q - train_mean

    return support / torch.norm(input=support, p=2, dim=1, keepdim=True), query / torch.norm(input=query, p=2, dim=1, keepdim=True)

def torch_diag_block_matrix(a, b):
    return torch.cat([
        torch.cat([a, torch.zeros(a.shape[0], b.shape[1])], dim=1),
        torch.cat([torch.zeros(b.shape[0], a.shape[1]), b], dim=1)
        ], dim=0)

def save_pickle(file_name, data):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.DEFAULT_PROTOCOL)

def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def torch_copy_and_add(base_instance, copy_instance, copy_scale):
    if len(copy_instance.shape) == 1:
        return torch.cat([base_instance, torch.stack([copy_instance] * copy_scale)])
    else:
        return torch.cat([base_instance] + [copy_instance] * copy_scale)

def str_filter_int(s):
    # 仅允许 '_x' ~ 在此之后出现的第一个'_' 之间的数字.
    if '_x' in s:
        # return int("".join(list(filter(str.isdigit, s[s.find('_x'):s.find('_', s.find('_x') + 1)]))))
        return int(s[s.find('_x')+2 : s.find('_', s.find('_x')+1)])
    return 0

def cpnt_extd_plot(x, y, plot_axis, ci=None, plot_title='', plot_label=['', ''], logger=None, plot_file_name=None, yaml_file_name=None):
    if yaml_file_name is not None:
        cpnt_extd_plot_params = {
            'x': x,
            'y': [float(v) for v in y],
            'plot_axis': plot_axis,
            'ci': [float(v) for v in ci],
            'plot_title': plot_title,
            'plot_label': plot_label,
            'yaml_file_name': yaml_file_name
            }
        with open(yaml_file_name, 'w') as f:
            yaml.dump(cpnt_extd_plot_params, f)

    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.use('Agg')

    plt.plot(x, y, 'o-', label=plot_title)
    plt.fill_between(x, [i - j for i, j in zip(y, ci)], [i + j for i, j in zip(y, ci)], alpha=.2)
    plt.axis(plot_axis)
    plt.xlabel(plot_label[0])
    plt.ylabel(plot_label[1])
    max_idx = y.index(max(y)) # 仅标出最大值.
    plt.annotate('%.2f' % (y[max_idx]), xy=(x[max_idx], y[max_idx]))
    plt.legend()
    if plot_file_name is not None:
        plt.savefig(plot_file_name)
        plt.clf()
    if logger is not None:
        logger.info(f'Result Plot x:\n    {x}')
        logger.info(f'Result Plot y:\n    {y}')
        logger.info(f'Result Plot is saved successfully.')

def sorted_eig(m):
    ## Test it:
    # from numpy import linalg as LA
    # eigenValues, eigenVectors = LA.eig(sigma.cpu().detach().numpy())
    # idx = eigenValues.argsort()[::-1]   
    # eigenValues = eigenValues[idx]
    # eigenVectors = eigenVectors[:,idx]
    # print(eigenVectors)

    # m = m.pin_memory().cuda(non_blocking=True)
    s, u = torch.eig(m, eigenvectors=True) # torch.svd 返回的是降序特征根的, eig 则不是.
    s = s[:, 0] # 特征根都是实数.

    descend_idx = torch.argsort(s.abs(), descending=True)

    return s[descend_idx], u[:, descend_idx]

def get_transform_matrices_handle(x, mean, n_components=None, ratio=None, use_corr=True):
    # 双层函数, 请先调用 get_transform_matrices_handle 来初始化 (不同mtype操作所需的矩阵).
    # x: 为各种协方差、相关系数矩阵的基础矩阵.

    # x: [size, feat_dim]
    # mean: 训练集的.
    # ? sigma 的 n 用什么的, 这里用的是size, 一个batch里面的数量.
    sigma = torch_cov(x, mean)
    # sigma_debug = torch.from_numpy(np.cov(x.cpu().detach().numpy().T)).to(torch.device('cuda'))
    # print(torch.norm(sigma - sigma_debug))

    # sigma = sigma.double()

    s, u = sorted_eig(sigma)

    # 与 sklearn完全一致的pca:
    u_svd, s_svd, v_svd = torch.svd(x - mean)

    def svd_flip(u_cur, v_cur):
        max_abs_cols = torch.argmax(torch.abs(u_cur), dim=0)
        signs = torch.sign(u_cur[max_abs_cols, range(u_cur.shape[1])])
        u_cur *= signs
        v_cur *= signs.unsqueeze(-1)
        return v_cur
    v_svd = svd_flip(u_svd, v_svd.t())


    if n_components is None and ratio is not None:
        # 切割特征向量, 根据ratio.
        def split_by_ratio(s_r, u_r):
            for i_split in range(s_r.shape[0]-1, -1, -1):
                if s_r[0:i_split].sum() / s_r.sum() < ratio:
                    break
            s_r = s_r[0:(i_split + 1)]
            u_r = u_r[:, 0:(i_split + 1)]
            return s_r, u_r
        s, u = split_by_ratio(s, u)
        s_svd, u_svd = split_by_ratio(s_svd, u_svd)
    elif n_components is not None:
        # 切割特征向量, 根据n_components, 且此时优先级更高.
        s = s[0:n_components]
        u = u[:, 0:n_components]

        v_svd = v_svd[0:n_components]
    else:
        # 不切割, 等后面expt来切.
        pass

    s_minus_half = torch.diag(1 / torch.sqrt(s + 1e-5))

    if use_corr:
        d = torch.diag(sigma) + 1e-5
        stddev = torch.sqrt(d)
        corr = sigma.div(stddev.expand_as(sigma))
        corr = corr.div(torch.transpose(stddev.expand_as(sigma), -2, -1))

        # clamp between -1 and 1
        # probably not necessary but numpy does it
        corr = torch.clamp(corr, -1.0, 1.0)

        sigma_diag_sqrt = 1 / torch.sqrt(torch.diag(sigma) + 1e-5)
        v_minus_half = torch.diag(sigma_diag_sqrt)

        theta, g = torch.eig(corr, eigenvectors=True)
        theta = theta[:, 0]
        theta_minus_half = torch.diag(1 / torch.sqrt(theta + 1e-5))

    def core(mtype):
        if 'zca_wtg' in mtype:
            # TODO: 这里乘法的顺序待确定.
            return u @ s_minus_half @ torch.transpose(u, -2, -1)
        elif 'zca_cpnt_wtg' in mtype:
            n_components_str = 'zca_cpnt_wtg_'
            n_components_zca_wtg = int(mtype[mtype.find(n_components_str)+len(n_components_str) : mtype.find('__prt')])
            return u[:, 0 : n_components_zca_wtg] @ s_minus_half[0 : n_components_zca_wtg, 0 : n_components_zca_wtg] @ torch.transpose(u[:, 0 : n_components_zca_wtg], -2, -1)

        elif 'pca_wtg' in mtype:
            return u @ s_minus_half
        elif 'pca_cpnt_wtg' in mtype:
            n_components_str = 'pca_cpnt_wtg_'
            n_components_wtg = int(mtype[mtype.find(n_components_str)+len(n_components_str) : mtype.find('__prt')])
            return u[:, 0 : n_components_wtg] @ s_minus_half[0 : n_components_wtg, 0 : n_components_wtg]

        elif 'pca_pure' in mtype:
            # => pca.transform(x)
            return u
        elif 'pca_cpnt_pure' in mtype:
            n_components_str = 'pca_cpnt_pure_'
            return u[:, 0 : int(mtype[mtype.find(n_components_str)+len(n_components_str) : mtype.find('__prt')])]

        elif 'pca_svd' in mtype:
            return v_svd.t()
        elif 'pca_cpnt_svd' in mtype:
            # pca's n_component in norm_type(mtype):
            return v_svd[0 : str_filter_int(mtype)].t()

        elif 'zca_corr_wtg' in mtype:
            corr_minus_half = g @ theta_minus_half @ torch.transpose(g, -2, -1)
            return corr_minus_half @ v_minus_half
        elif 'pca_corr_wtg' in mtype:
            return theta_minus_half @ torch.transpose(g, -2, -1) @ v_minus_half
        else:
            raise Exception(f'mtype: {mtype}.\n' + 'Oops! get_transform_matrices_handle if-else error.')
    return core

# def get_transform_matrices_handle_batch(x, mean):
#     sigma = torch_cov(x, mean)
#     # sigma = sigma.double()

#     def batch_eig(mat):
#         def sorted_eig(m):
#             s, u = torch.eig(m, eigenvectors=True)
#             u, s = u.t(), s[:, 0]
#             descend_idx = torch.argsort(s, descending=True)
#             return s, u[descend_idx, :]
#         s_batch_list, u_batch_list = [], []
#         for batch in mat:
#             s_tmp, u_tmp = sorted_eig(batch)
#             s_batch_list.append(s_tmp)
#             u_batch_list.append(u_tmp)
#         return torch.stack(s_batch_list, dim=0), torch.stack(u_batch_list, dim=0)

#     s, u = batch_eig(sigma)
#     s_minus_half = torch.diag_embed(1 / torch.sqrt(s + 1e-5))

#     corr = sigma.squeeze(0)
#     d = torch.diagonal(sigma, dim1=1, dim2=2) + 1e-5

#     stddev = torch.sqrt(d)
#     corr = sigma.div(stddev.expand_as(sigma))
#     corr = corr.div(torch.transpose(stddev.expand_as(sigma), -2, -1))

#     corr = torch.clamp(corr, -1.0, 1.0)

#     # 注意, 这里diag操作可能有点问题:
#     sigma_diag_sqrt = 1 / torch.sqrt(torch.diagonal(sigma, dim1=1, dim2=2) + 1e-5)
#     v_minus_half = torch.diag_embed(sigma_diag_sqrt)

#     theta, g = batch_eig(corr)
#     theta_minus_half = torch.diag_embed(1 / torch.sqrt(theta + 1e-5))

#     def core(mtype):
#         if 'zca_whitening' in mtype:
#             return u @ s_minus_half @ torch.transpose(u, -2, -1)
#         elif 'pca_whitening' in mtype:
#             return s_minus_half @ torch.transpose(u, -2, -1)
#         elif 'pca_pure' in mtype:
#             return torch.transpose(u, -2, -1)
#         elif 'zca_corr_whitening' in mtype:
#             corr_minus_half = g @ theta_minus_half @ torch.transpose(g, -2, -1)
#             return corr_minus_half @ v_minus_half
#         elif 'pca_corr_whitening' in mtype:
#             return theta_minus_half @ torch.transpose(g, -2, -1) @ v_minus_half
#     return core

def get_lr(optimizer):
    lr = ['{:.6f}'.format(param_group['lr']) for param_group in optimizer.param_groups]

    return ','.join(lr)

def difference_set_of_list(l1, l2):
    return list(set(l1) - set(l2))

def create_affinity(X, knn):
    N, D = X.shape

    nbrs = NearestNeighbors(n_neighbors=knn).fit(X)
    dist, knnind = nbrs.kneighbors(X)

    row = np.repeat(range(N), knn - 1)
    col = knnind[:, 1:].flatten()
    data = np.ones(X.shape[0] * (knn - 1))
    W = sparse.csc_matrix((data, (row, col)), shape=(N, N), dtype=np.float)

    return W

def alchemy_log(content, log_file_name='/home/zhangyk/Few-shot-Framework/run/alchemy_sequential.log'):
    with open(log_file_name, 'a') as f:
        f.write(content + '\n')


def update_params(loss, params, step_size, weight_decay, momentum, mom_buffer, first_order=True, **kwargs):
    name_list, tensor_list = zip(*params.items())

    grads = torch.autograd.grad(loss, tensor_list, create_graph=not first_order)
    updated_params = OrderedDict()

    for name, param, grad in zip(name_list, tensor_list, grads):
        grad = grad + weight_decay * param
        grad = grad + momentum * mom_buffer[name]
        mom_buffer[name] = grad

        updated_params[name] = param - step_size * grad

    return updated_params, mom_buffer