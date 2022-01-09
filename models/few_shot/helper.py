from pickle import load
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader

from models.utils import pretrain_prepare_batch, difference_set_of_list

# from models.dataloader.mini_imagenet import MiniImageNet
# from models.dataloader.tiered_imagenet import TieredImageNet

from models.dataloader.lrw import LRW
from models.dataloader.lrw1000 import LRW1000

from models.few_shot.protonet import ProtoNet
# from models.few_shot.matchingnet import MatchingNet
# from models.few_shot.relationnet import RelationNet
# from models.few_shot.metaoptnet import MetaOptNet
# from models.few_shot.maml import MAML
# from models.few_shot.linear import Linear
# from models.few_shot.feat import FEAT
# from models.few_shot.protonet import LaplacianProtoNet
# from models.few_shot.efficientnet import EfficientNet
# from models.few_shot.weijingnet import WeijingNet
from models.few_shot.rgmaml import MultiModalMAML

from models.few_shot.protonet_mm import MultiModalProtoNet
# from models.few_shot.tima import MultiModalTIMA
# from models.few_shot.tima_plus import MultiModalTIMAPlus
# from models.few_shot.rgprotonet import RGMultiModalProtoNet
# from models.few_shot.rgtima import RGTIMA
# from models.few_shot.matchingnet_mm import MultiModalMatchingNet
# from models.few_shot.rgtima_plus import RGTIMAPlus
# from models.few_shot.castle import CASTLE
# from models.few_shot.rgprotonet_plus import RGMultiModalProtoNetPlus
# from models.few_shot.rgprotonet_plus_wo_attn import RGMultiModalProtoNetPlusWOATTN
# from models.few_shot.rgprotonet_spp_attn import RGMultiModalProtoNetSPPATTN
# from models.few_shot.rgprotonet_plus_crs import RGMultiModalProtoNetPlusCRS
from models.few_shot.protocat import ProtoCAT
from models.few_shot.protocat_plus import ProtoCAT_Plus

# from models.few_shot.rouge import Rouge

# from models.classical.linearclassifier import LinearClassifier

# from models.few_shot.protonet import ProtoNetPretrainClassifier
from models.few_shot.protonet import CRGPretrainClassifier
from models.few_shot.protonet import RTPretrainClassifier

from torch.cuda.amp import autocast, GradScaler

from models.sampler import RandomSampler

# from models.utils import check_model_changed_handle

import os.path as osp

PRETRAIN_FILE_NAME_SUFFIX_0 = '-pre.pth'
PRETRAIN_FILE_NAME_SUFFIX_1 = '-pre.pt'
PRETRAIN_FILE_NAME_SUFFIX_2 = '-pre.pth.tar'

class PrepareFunc(object):
    def __init__(self, args):
        self.args = args
    def prepare_model(self, train_num_classes):
        # 这里决定了是什么模型.
        model = eval(self.args.model_class)(self.args, train_num_classes)

        # load pre-trained model (no FC weights 注意只加载backbone的参数.)
        if self.args.init_weights is not None:
            print(f'Loading the pre-training model...')

            # load 核心函数:
            def load_weights(cur_backbone_class, cur_model):
                model_dict = cur_model.state_dict()

                if osp.isfile(self.args.init_weights):
                    pretrained_dict = torch.load(self.args.init_weights)
                elif osp.isfile(osp.join(self.args.init_weights, cur_backbone_class + PRETRAIN_FILE_NAME_SUFFIX_0)):
                    try:
                        pretrained_dict = torch.load(osp.join(self.args.init_weights, cur_backbone_class + PRETRAIN_FILE_NAME_SUFFIX_0))['params']
                    except:
                        pretrained_dict = torch.load(osp.join(self.args.init_weights, cur_backbone_class + PRETRAIN_FILE_NAME_SUFFIX_0))
                elif osp.isfile(osp.join(self.args.init_weights, cur_backbone_class + PRETRAIN_FILE_NAME_SUFFIX_1)):
                    pretrained_dict = torch.load(osp.join(self.args.init_weights, cur_backbone_class + PRETRAIN_FILE_NAME_SUFFIX_1), map_location=torch.device('cpu')).get('video_model')
                elif osp.isfile(osp.join(self.args.init_weights, cur_backbone_class + PRETRAIN_FILE_NAME_SUFFIX_2)):
                    pretrained_dict = torch.load(osp.join(self.args.init_weights, cur_backbone_class + PRETRAIN_FILE_NAME_SUFFIX_2))["model_state_dict"]
                else:
                    raise Exception(f'Loading pretrained model error: file not exists. [cur_backbone_class: {cur_backbone_class}]')

                def fine_tuning_params_dict(cur_d, modify_str=None, option='flt'):
                    # 为了对应 model_dict.keys(), 微调一下pretrained模型的key.
                    if option == 'add':
                        ret_d = {modify_str + k: v for k, v in cur_d.items() if modify_str + k in model_dict}
                        del_keys = [k for k in cur_d.keys() if modify_str + k not in model_dict]
                    elif option == 'del':
                        ret_d = {k.replace(modify_str, ''): v for k, v in cur_d.items() if k.replace(modify_str, '') in model_dict}
                        del_keys = [k for k in cur_d.keys() if k.replace(modify_str, '') not in model_dict]
                    elif option == 'flt':
                        ret_d = {k: v for k, v in cur_d.items() if k in model_dict}
                        del_keys = [k for k in cur_d.keys() if k not in model_dict]
                    if len(del_keys) != 0:
                        print(f'The deleted pre-trained\'s keys (IN pre-trained model BUT NOT IN current): {del_keys}.')

                    return ret_d

                def substr_in_keys(cur_d, cur_s):
                    # 只要存在一个有就return True.
                    for i in cur_d.keys():
                        if cur_s in i:
                            return True
                    return False

                # 仅能在pretrained_dict中改key:
                # 现有模型存在'encoder.encoder.', 但是加载的pretrain模型仅有一个'encoder.'
                if substr_in_keys(model_dict, 'encoder.encoder') and \
                    (not substr_in_keys(pretrained_dict, 'encoder.encoder') and substr_in_keys(pretrained_dict, 'encoder')):
                    pretrained_dict = fine_tuning_params_dict(pretrained_dict, 'encoder.', 'add')
                # 现有模型没有'encoder.', 但是加载的pretrain模型有.
                if not substr_in_keys(model_dict, 'encoder') and substr_in_keys(pretrained_dict, 'encoder'):
                    pretrained_dict = fine_tuning_params_dict(pretrained_dict, 'encoder.', 'del')
                # 现有模型没有'module.', 但是加载的pretrain模型有.
                if not substr_in_keys(model_dict, 'module') and substr_in_keys(pretrained_dict, 'module'):
                    pretrained_dict = fine_tuning_params_dict(pretrained_dict, 'module.', 'del')
                else:
                    pretrained_dict = fine_tuning_params_dict(pretrained_dict)

                if len(pretrained_dict.keys()) == 0:
                    raise Exception(f'{cur_backbone_class}: Oops! Failed to load pre-trained model.')
                elif len(pretrained_dict.keys()) == len(model_dict.keys()):
                    print(f'{cur_backbone_class}: All pre-trained model parameters are loaded.')
                else:
                    print(f'{cur_backbone_class}: Missing loading pre-training model parameters (IN current model BUT NOT IN pre-trained) {[i for i in model_dict.keys() if i not in pretrained_dict.keys()]}.')

                model_dict.update(pretrained_dict)
                cur_model.load_state_dict(model_dict)

            # load 核心函数结束.
            if self.args.unimodal or osp.isfile(self.args.init_weights):
                # 单模态情况: unimodal 参数为True, 或者init_weights不是路径, 而是文件.
                if self.args.unimodal:
                    if self.args.dataset in ['LRW', 'LRW1000']:
                        load_weights(f'{self.args.backbone_class}{self.args.backend_type[0]}_{self.args.dataset}', model)
                    else:
                        load_weights(f'{self.args.backbone_class}{self.args.backend_type[0]}', model)
                else:
                    load_weights(self.args.backbone_class, model)
            else:
                for bkb, mdl in zip(self.args.backbone_class, self.args.mm_list):
                    load_weights(f'{bkb}{self.args.backend_type[self.args.mm_list.index(mdl)]}_{self.args.dataset}', eval(f'model.encoder_{mdl}'))

            # check_model_changed_handle.set_model(model)

        if torch.cuda.is_available():
            model = model.to(torch.device('cuda'))
            if ',' in self.args.gpu:
                print('Using nn.DataParallel...')
                if self.args.unimodal:
                    model.encoder = nn.DataParallel(model.encoder)
                else:
                    for i in self.args.mm_list:
                        exec(f'model.encoder_{i} = nn.DataParallel(model.encoder_{i})')

        else:
            raise Exception('Oops! Currently it does not support CPU training.')

        return model

    def prepare_optimizer(self, model):
        top_para = [v for k,v in model.named_parameters() if ('encoder' not in k and 'args' not in k)]
        top_params_keys = [k for k,v in model.named_parameters() if ('encoder' not in k and 'args' not in k)]
        if len(top_params_keys) != 0:
            print(f'Top-level parameters (other than Backbone): {top_params_keys}')
        # as in the literature, we use ADAM for ConvNet and SGD for other backbones
        if self.args.meta:
            optimizer = optim.Adam(
                [{'params': top_para, 'lr': self.args.lr * self.args.lr_mul}],
                lr=self.args.lr,
                # weight_decay=args.weight_decay, do not use weight_decay here
            )
        else:
            def set_optimizer(backbone_class, cur_encoder):
                if backbone_class == 'ConvNet':
                    return optim.Adam(
                        [{'params': cur_encoder.parameters()},
                        {'params': top_para, 'lr': self.args.lr * self.args.lr_mul}],
                        lr=self.args.lr,
                        # weight_decay=args.weight_decay, do not use weight_decay here
                        )
                elif backbone_class == 'Conv3dResNet' or backbone_class == 'Conv1dResNet':
                    return optim.Adam(
                        [{'params': cur_encoder.parameters()},
                        {'params': top_para, 'lr': self.args.lr * self.args.lr_mul}],
                        lr=self.args.lr,
                        weight_decay=1e-4
                        )
                elif backbone_class == 'MLGCN':
                    return optim.SGD(
                        model.get_config_optim(self.args.lr, self.args.lr_mul),
                        lr=self.args.lr,
                        momentum=self.args.momentum,
                        weight_decay=self.args.weight_decay
                        )
                elif backbone_class == 'MetaLearner':
                    return optim.Adam(
                        [{'params': cur_encoder.parameters()}],
                        lr=self.args.lr
                        )
                else:
                    return optim.SGD(
                        [{'params': cur_encoder.parameters()},
                        {'params': top_para, 'lr': self.args.lr * self.args.lr_mul}],
                        lr=self.args.lr,
                        momentum=self.args.momentum,
                        nesterov=True,
                        weight_decay=self.args.weight_decay
                        )

            if self.args.unimodal:
                optimizer = set_optimizer(self.args.backbone_class, model.encoder)
            else:
                optimizer = {}
                for i, j in zip(self.args.mm_list, self.args.backbone_class):
                    optimizer[i] = set_optimizer(j, eval(f'model.encoder_{i}'))

        # trick
        # 关注step_size等参数.
        def set_lr_scheduler(cur_type, optmz):
            if cur_type == 'step':
                return optim.lr_scheduler.StepLR(
                    optmz,
                    step_size=int(self.args.step_size),
                    gamma=self.args.gamma
                    )
            elif cur_type == 'multistep':
                return optim.lr_scheduler.MultiStepLR(
                    optmz,
                    milestones=[int(_) for _ in self.args.step_size.split(',')],
                    gamma=self.args.gamma,
                    )
            elif cur_type == 'cosine':
                return optim.lr_scheduler.CosineAnnealingLR(
                    optmz,
                    self.args.max_epoch,
                    eta_min=self.args.cosine_annealing_lr_eta_min   # a tuning parameter
                    )
            elif cur_type == 'plateau':
                return optim.lr_scheduler.ReduceLROnPlateau(
                    optmz,
                    mode='min',
                    factor=self.args.gamma,
                    patience=5
                    )
            else:
                raise ValueError('No Such Scheduler')

        if self.args.unimodal:
            lr_scheduler = set_lr_scheduler(self.args.lr_scheduler, optimizer)
        else:
            lr_scheduler = {}
            for i, j in zip(self.args.mm_list, self.args.lr_scheduler):
                lr_scheduler[i] = set_lr_scheduler(j, optimizer[i])

        if self.args.grad_scaler:
            scaler = GradScaler()
        else:
            scaler = None
        return optimizer, lr_scheduler, scaler

    def prepare_loss_fn(self):
        if self.args.loss_fn == 'F-cross_entropy':
            return F.cross_entropy
        elif self.args.loss_fn == 'nn-cross_entropy':
            return nn.CrossEntropyLoss().to(torch.device('cuda'))
        elif self.args.loss_fn == 'nn-nll':
            return nn.NLLLoss()
        elif self.args.loss_fn == 'nn-mse':
            return nn.MSELoss()
        elif self.args.loss_fn == 'nn-multi_label_soft_margin':
            return nn.MultiLabelSoftMarginLoss()
        # else:
        #     raise Exception('prepare_loss_fn error: self.args.loss_fn not exists.')

    def prepare_dataloader(self, option=0b111, is_only_one=False, rougee_flag=None, is_weijing_hc=False, gfsl_test=False, gfsl_train=False):
        def taskloader(stype, args, shot, way, query, episodes_per_epoch):
            dataset = eval(args.dataset)(stype, args)
            task_loader = DataLoader(
                dataset,
                batch_sampler=dataset.sampler_handle(
                    dataset=dataset,
                    episodes_per_epoch=episodes_per_epoch,
                    shot=shot,
                    way=way,
                    query=query,
                    num_tasks=args.meta_batch_size,
                    ),
                num_workers=args.num_workers,
                pin_memory=True
                )
            return task_loader, dataset.num_classes

        def pretrainloader(stype, args, shuffle=True):
            dataset = eval(args.dataset)(stype, args)
            pretrain_loader = DataLoader(
                dataset=dataset,
                batch_size=args.batch_size,
                shuffle=shuffle,
                num_workers=args.num_workers,
                pin_memory=True
            )
            return pretrain_loader, dataset.num_classes

        def gfslloader(stype, args, way, query, episodes_per_epoch):
            meta_test_base_set = eval(args.dataset)(stype, args)

            many_shot_sampler = RandomSampler(
                meta_test_base_set.label,
                episodes_per_epoch,
                way * query
                )
            gfsl_loader = DataLoader(
                dataset=meta_test_base_set,
                batch_sampler=many_shot_sampler,
                num_workers=args.num_workers,
                pin_memory=True
                )
            return gfsl_loader, meta_test_base_set.num_classes

        args = self.args

        if gfsl_train:
            return (gfslloader('train', args, args.train_way, args.train_query, args.episodes_per_train_epoch)), (None, None), (None, None)
        if gfsl_test:
            return (None, None), \
                (gfslloader('aux_val', args, args.val_way, args.val_query, args.episodes_per_val_epoch)), \
                    (gfslloader('aux_test', args, args.test_way, args.val_query, args.episodes_per_test_epoch))

        if is_weijing_hc:
            return (None, None), \
                (taskloader('train', args, args.val_shot, args.train_way - args.val_way, args.val_query, args.episodes_per_val_epoch)), \
                    (taskloader('train', args, args.test_shot, args.train_way - args.test_way, args.test_query, args.episodes_per_test_epoch)) \

        if rougee_flag == 'mean':
            return (pretrainloader('train', args, False)), (None, None), (None, None)
        elif rougee_flag == 'test':
            return (None, None), (None, None), (taskloader('test', args, args.test_shot, args.test_way, args.test_query, args.rougee_episodes_per_epoch))

        if is_only_one:
            if option & 0b100:
                return (pretrainloader('train', args)), (None, None), (None, None)
            elif option & 0b010:
                return (None, None), (pretrainloader('val', args)), (None, None)
            elif option & 0b001:
                return (None, None), (None, None), (pretrainloader('test', args))
            else:
                raise Exception('Error using function parameters.')

        train_taskloader, train_num_classes = taskloader('train', args, args.train_shot, args.train_way, args.train_query, args.episodes_per_train_epoch) \
            if option & 0b100 else pretrainloader('train', args)
        val_taskloader, val_num_classes = taskloader('val', args, args.val_shot, args.val_way, args.val_query, args.episodes_per_val_epoch) \
            if option & 0b010 else pretrainloader('val', args)
        test_taskloader, test_num_classes = taskloader('test', args, args.test_shot, args.test_way, args.test_query, args.episodes_per_test_epoch) \
            if option & 0b001 else pretrainloader('test', args)

        # return {'train': train_taskloader, 'val': val_taskloader, 'test': test_taskloader}
        return (train_taskloader, train_num_classes), (val_taskloader, val_num_classes), (test_taskloader, test_num_classes)

    def prepare_prepare_batch_func(self, model_prepare_batch, option):
        args = self.args
        train_prepare_batch = model_prepare_batch(args.train_way, args.train_query, args.train_shot, args.meta_batch_size) \
            if option & 0b100 else pretrain_prepare_batch
        val_prepare_batch = model_prepare_batch(args.val_way, args.val_query, args.val_shot, args.meta_batch_size) \
            if option & 0b010 else pretrain_prepare_batch
        test_prepare_batch = model_prepare_batch(args.test_way, args.test_query, args.test_shot, args.meta_batch_size) \
            if option & 0b001 else pretrain_prepare_batch

        return train_prepare_batch, val_prepare_batch, test_prepare_batch

