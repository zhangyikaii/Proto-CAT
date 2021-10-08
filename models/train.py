import torch
from torch.optim import Optimizer
from torch.nn import Module
from torch.utils.data import DataLoader

from typing import Callable, List, Union
import os
import os.path as osp
from tqdm import tqdm
from collections import OrderedDict

from models.callbacks import (
    ProgressBarLogger,
    CallbackList,
    Callback,
    EvaluateFewShot,
    CSVLogger
)

from models.few_shot.protonet import fit_handle as ProtoNet_fit_handle
# from models.few_shot.matchingnet import fit_handle as MatchingNet_fit_handle
# from models.few_shot.relationnet import fit_handle as RelationNet_fit_handle
# from models.few_shot.metaoptnet import fit_handle as MetaOptNet_fit_handle
# from models.few_shot.maml import fit_handle as Maml_fit_handle
# Linear_fit_handle = ProtoNet_fit_handle
# FEAT_fit_handle = ProtoNet_fit_handle
# Rouge_fit_handle = ProtoNet_fit_handle
# from models.few_shot.protonet import laplacian_protonet_fit_handle as LaplacianProtoNet_fit_handle
# from models.few_shot.efficientnet import fit_handle as EfficientNet_fit_handle
# from models.few_shot.weijingnet import fit_handle as WeijingNet_fit_handle

from models.few_shot.protonet_mm import fit_handle as MultiModalProtoNet_fit_handle
# from models.few_shot.tima import fit_handle as MultiModalTIMA_fit_handle
# from models.few_shot.tima_plus import fit_handle as MultiModalTIMAPlus_fit_handle
# from models.few_shot.rgprotonet import fit_handle as RGMultiModalProtoNet_fit_handle
# from models.few_shot.rgtima import fit_handle as RGTIMA_fit_handle
# from models.few_shot.matchingnet_mm import fit_handle as MultiModalMatchingNet_fit_handle
# from models.few_shot.rgtima_plus import fit_handle as RGTIMAPlus_fit_handle
# from models.few_shot.castle import fit_handle as CASTLE_fit_handle
# RGMultiModalProtoNetPlus_fit_handle = CASTLE_fit_handle
# RGMultiModalProtoNetPlusWOATTN_fit_handle = RGMultiModalProtoNetPlus_fit_handle
# RGMultiModalProtoNetSPPATTN_fit_handle = RGMultiModalProtoNet_fit_handle
# from models.few_shot.rgprotonet_plus_crs import fit_handle as RGMultiModalProtoNetPlusCRS_fit_handle
from models.few_shot.protocat import fit_handle as ProtoCAT_fit_handle
ProtoCAT_Plus_fit_handle = ProtoCAT_fit_handle

# from models.few_shot.protonet import protonet_pretrain_fit_handle as ProtoNetPretrainClassifier_fit_handle
from models.few_shot.protonet import crg_pretrain_fit_handle as CRGPretrainClassifier_fit_handle
RTPretrainClassifier_fit_handle = CRGPretrainClassifier_fit_handle

# from models.classical.linearclassifier import fit_handle as LinearClassifier_fit_handle

from models.few_shot.helper import PrepareFunc

from models.utils import get_lr, set_logger, pretrain_prepare_batch, multimodal_pretrain_prepare_batch, set_gpu, set_seeds, load_pickle, save_pickle
from models.metrics import metrics_handle, ROOT_PATH

# from models.ycy import RouGee

class Trainer(object):
    def __init__(self, args):
        self.gpu = args.gpu
        set_gpu(self.gpu, args.gpu_space_require)
        set_seeds(torch_seed=args.torch_seed, cuda_seed=args.cuda_seed, np_seed=args.np_seed, random_seed=args.random_seed)

        torch.autograd.set_detect_anomaly(True)

        self.logger_filename = ROOT_PATH + f'{args.logger_filename}/process/{args.params_str}.log'
        self.result_filename = ROOT_PATH + f'{args.logger_filename}/result/{args.params_str}.csv'

        self.logger = set_logger(self.logger_filename, 'train_logger')
        for k in sorted(vars(args).keys()):
            self.logger.info(k + ': %s' % str(vars(args)[k]))

        prepare_handle = PrepareFunc(args)
        # 前向传播所需:
        # ( model 有且仅有这一个, callbacks 基类派生类都是共享这一个model. 下面其他东西也是这样存储 )
        """
        准备 Dataloader
        """
        (self.train_loader, train_num_classes), (self.val_loader, _), (self.test_loader, _) = \
            prepare_handle.prepare_dataloader(option=args.paradigm)

        """
        准备 Model, Optimizer, loss_fn, callbacks
        """
        self.model = prepare_handle.prepare_model(train_num_classes)

        self.optimizer, self.lr_scheduler, self.scaler = prepare_handle.prepare_optimizer(self.model)
        self.loss_fn = prepare_handle.prepare_loss_fn()
        self.fit_handle = eval(args.model_class + '_fit_handle')(
            model=self.model,
            optimizer=self.optimizer,
            scaler=self.scaler,
            loss_fn=self.loss_fn,
            gfsl_test=args.gfsl_test
            )

        # 接下来要准备fit函数之前的所有东西, 包括callbacks.
        # 记录数据所需:
        # 这里的params统一传到基类成员, 所有派生类共享. 注意这里一定要精简.

        # # 目前pretrain的prepare_batch_func不支持多模态.
        # self.train_prepare_batch, val_prepare_batch, test_prepare_batch = prepare_handle.prepare_prepare_batch_func(
        #     model_prepare_batch=self.model.multimodal_prepare_kshot_task if ',' in args.multimodal_option else self.model.prepare_kshot_task,
        #     option=0b111 if not args.pretrain_mode else 0b011
        # )
        # self.train_prepare_batch, val_prepare_batch, test_prepare_batch = prepare_handle.prepare_prepare_batch_func(
        #     model_prepare_batch=self.model.prepare_kshot_task,
        #     option=args.paradigm,
        #     query_is_label=True if args.model_class != 'Linear' else False
        # )

        self.train_prepare_batch, val_prepare_batch, test_prepare_batch = prepare_handle.prepare_prepare_batch_func(
            model_prepare_batch=self.model.prepare_kshot_task,
            option=args.paradigm
            )

        DEBUG_CLASSICAL_ML = False
        if DEBUG_CLASSICAL_ML:
            (self.train_loader, train_num_classes), (self.val_loader, _), (self.test_loader, _) = \
                prepare_handle.prepare_dataloader(option=0b000)
            self.fit_handle = ProtoNet_fit_handle(
                model=self.model,
                optimizer=self.optimizer,
                scaler=self.scaler,
                loss_fn=self.loss_fn
                )
            test_prepare_batch = pretrain_prepare_batch

        self.verbose, self.epoch_verbose = args.verbose, args.epoch_verbose
        # self.params = {
        #     'max_epoch': args.max_epoch,
        #     'verbose': args.verbose,
        #     'metrics': (self.metrics or []),
        #     'prepare_batch': prepare_kshot_task(args.test_way, args.test_query),
        #     'loss_fn': self.loss_fn,
        #     'optimizer': self.optimizer,
        #     'lr_scheduler': self.lr_scheduler
        # }

        # args 是一个参数集合, 期望在这里分模块, 对每个类 对应特定的功能, 类的参数也要**具体化**, 这样才可以一层层分解, 较好.

        self.model_filepath = args.model_filepath
        self.train_monitor = args.train_monitor
        self.batch_metrics = metrics_handle(self.model, self.train_monitor)
        assist_metrics = metrics_handle(self.model, 'categorical accuracy per class') if args.acc_per_class else None

        self.do_train = args.do_train
        self.do_test = args.do_test

        self.max_epoch = args.max_epoch

        self.gfsl_train = args.gfsl_train
        if args.gfsl_train:
            (self.cooperative_query_dataloader, _), (_, _), (_, _) = prepare_handle.prepare_dataloader(gfsl_train=True)

        if args.weijing_is_hc:
            (_, _), (val_base_loader, _), (test_base_loader, _) = prepare_handle.prepare_dataloader(is_weijing_hc=True)
        elif args.gfsl_test:
            (_, _), (val_base_loader, _), (test_base_loader, _) = prepare_handle.prepare_dataloader(gfsl_test=True)
        else:
            val_base_loader, test_base_loader = None, None
        self.evaluate_handle = EvaluateFewShot(
            val_loader=self.val_loader,
            test_loader=self.test_loader,
            val_prepare_batch=val_prepare_batch,
            test_prepare_batch=test_prepare_batch,
            batch_metrics=self.batch_metrics,
            assist_metrics=assist_metrics,
            eval_fn=self.fit_handle,
            test_interval=args.test_interval,
            max_epoch=self.max_epoch,
            model_filepath=args.model_filepath,
            monitor=self.train_monitor,
            save_best_only=True,
            mode='max',
            simulation_test=False,
            verbose=args.verbose,
            epoch_verbose=args.epoch_verbose,
            val_base_loader=val_base_loader,
            test_base_loader=test_base_loader,
            is_hc=args.weijing_is_hc,
            is_gfsl=args.gfsl_test
            )

        callbacks = [
            self.evaluate_handle,
            CSVLogger(
                self.result_filename,
                separator=',',
                append=False
                )
            ]

        if self.do_train:
            callbacks.append(
                ProgressBarLogger(
                    length=len(self.train_loader),
                    avg_monitor=self.train_monitor,
                    epoch_verbose=self.epoch_verbose
                    )
                )

        # LearningRateScheduler 最好直接在fit函数里面传一个lr_scheduler, 直接step吧. 看FEAT.
        self.callbacks = CallbackList((callbacks or []))
        self.callbacks.set_model_and_logger(self.model, self.logger)

        """
        meta
        """
        self.meta = args.meta

        # if args.init_weights is not None:
        #     self.evaluate_handle.predict_log(1, 'test_')

        self.mm_list = args.mm_list
        PREPROCESS_FEATURE_MEAN_STD = False
        if PREPROCESS_FEATURE_MEAN_STD:
            (feat_train_loader, feat_num_classes), (_, _), (_, _) = prepare_handle.prepare_dataloader(option=0b100, is_only_one=True)
            from models.dataloader.lrw import LRW_VIDEO_DATA_PATH_NAME, LRW_AUDIO_DATA_PATH_NAME
            FEATURE_MEAN_CACHE_FILE_NAME_SUFFIX = '_base_features_mean.pkl'
            FEATURE_STD_CACHE_FILE_NAME_SUFFIX = '_base_features_std.pkl'

            data_root_path_video = osp.join(args.data_path, LRW_VIDEO_DATA_PATH_NAME)
            data_root_path_audio = osp.join(args.data_path, LRW_AUDIO_DATA_PATH_NAME)

            cache_filepath = {
                'mean': {
                    'video': osp.join(data_root_path_video, '_'.join(args.backbone_class) + FEATURE_MEAN_CACHE_FILE_NAME_SUFFIX),
                    'audio': osp.join(data_root_path_audio, '_'.join(args.backbone_class) + FEATURE_MEAN_CACHE_FILE_NAME_SUFFIX)
                    },
                'std': {
                    'video': osp.join(data_root_path_video, '_'.join(args.backbone_class) + FEATURE_STD_CACHE_FILE_NAME_SUFFIX),
                    'audio': osp.join(data_root_path_audio, '_'.join(args.backbone_class) + FEATURE_STD_CACHE_FILE_NAME_SUFFIX)
                    }
                }

            self.model.eval()
            with torch.no_grad():
                def compute(cur_modal, cache_mean_filepath, cache_std_filepath):
                    if self.epoch_verbose:
                        cur_iter = enumerate(tqdm(feat_train_loader))
                    else:
                        cur_iter = enumerate(feat_train_loader)

                    data = {}
                    for batch_idx, batch in cur_iter:
                        x, y = multimodal_pretrain_prepare_batch(batch)
                        x = self.model.forward_get_feature(x, modal=cur_modal)

                        # 构造 类别 -> embedding(特征) 的字典, 为后续计算类中心打下基础.
                        tmp_dict = {} # <class ID (int), feature>
                        for i, v in enumerate(y):
                            v = v.item()
                            if v in tmp_dict.keys():
                                tmp_dict[v].append(x[cur_modal][i])
                            else:
                                tmp_dict[v] = [x[cur_modal][i]]
                        for i, v in tmp_dict.items():
                            tmp_dict[i] = torch.stack(v, dim=0)

                        # 该batch数据 加入data:
                        for i, v, in tmp_dict.items():
                            if i in data.keys():
                                data[i] = torch.cat([data[i], v.cpu().detach()], dim=0)
                            else:
                                data[i] = v.cpu().detach()

                    feat_train_feature = torch.cat([i for i in data.values()], dim=0)

                    save_pickle(cache_mean_filepath, torch.mean(feat_train_feature, dim=0, keepdim=True))
                    save_pickle(cache_std_filepath, torch.std(feat_train_feature, dim=0, unbiased=False, keepdim=True))

                compute('video', cache_filepath['mean']['video'], cache_filepath['std']['video'])
                compute('audio', cache_filepath['mean']['audio'], cache_filepath['std']['audio'])

                assert 0, 'PREPROCESS_FEATURE_MEAN_STD OK!'

        self.init_weights = args.init_weights
        self.unimodal = args.unimodal

    def delete_logs(self):
        os.remove(self.logger_filename)
        os.remove(self.result_filename)

    def lr_handle(self, epoch, epoch_logs):
        if self.unimodal:
            self.logger.info(f'lr: {get_lr(self.optimizer)}.')
            self.lr_scheduler.step()
        else:
            for mdl in self.mm_list:
                self.logger.info(f'lr: {get_lr(self.optimizer[mdl])} ({mdl}).')
                self.lr_scheduler[mdl].step()

    def test(self):
        # check_model_changed_handle.check_model(self.model)
        if self.do_test:
            if osp.isfile(self.model_filepath):
                # load from file.
                self.logger.info(f'Testing model: {self.model_filepath}')
                if self.verbose:
                    print(f'Testing model: {self.model_filepath}')
                state_dict = torch.load(self.model_filepath)

                if ',' not in self.gpu and 'module' in list(state_dict.keys())[0]:
                    # for models using nn.DataParallel:
                    print("WARNING: Loading test model in train.py test().")

                    tmp_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        tmp_state_dict[k.replace('.module', '')] = v # remove `.module`
                    state_dict = tmp_state_dict
                # elif ',' in self.gpu and 'module' not in list(state_dict.keys())[0]:
                #     raise Exception('Loading test model in train.py test(). The model may be training on a single GPU, but testing on multiple GPUs.')

                model_dict = self.model.state_dict()
                for pretrain_keys in ['v_cls.weight', 'v_cls.bias']:
                    if pretrain_keys in state_dict.keys() and pretrain_keys not in model_dict.keys():
                        del state_dict[pretrain_keys]

                if len(state_dict.keys()) != len(model_dict.keys()):
                    self.logger.info(f'Oops! Error loading model, not fully loaded: {[i for i in model_dict.keys() if i not in state_dict.keys()]}, {[i for i in state_dict.keys() if i not in model_dict.keys()]}.')
                    raise Exception(f'Oops! Error loading model, not fully loaded: {[i for i in model_dict.keys() if i not in state_dict.keys()]}, {[i for i in state_dict.keys() if i not in model_dict.keys()]}.')
                else:
                    self.model.load_state_dict(state_dict)
            else:
                self.logger.info(f'Testing model: {self.init_weights}.')
                if self.verbose:
                    print(f'Testing model: {self.init_weights}')

            # check_model_changed_handle.check_model(self.model)
            return self.evaluate_handle.predict_log(self.max_epoch, 'test_')
        return None

    def fit(self, call_test=False):
        print(f'Please check [do_train: {self.do_train}, do_test: {self.do_test}] again.')

        if not self.do_train:
            # # 两种情况: 1. 直接测试pretrain (init_weight非None), 2. 测试test_model_filepath的, 此时需要先加载模型.
            # # 直接测试, 要么从init_weights导入, 要么从test_model_filepath导入.
            return

        self.logger.info('Begin training...')
        if self.verbose:
            print('Begin training...')

        self.callbacks.on_train_begin()

        # from torch.utils.tensorboard import SummaryWriter
        # writer = SummaryWriter('runs/model')

        for epoch in range(1, self.max_epoch + 1):
            if self.gfsl_train:
                cur_iter = enumerate(zip(self.train_loader, self.cooperative_query_dataloader))
            else:
                cur_iter = enumerate(self.train_loader)

            self.callbacks.on_epoch_begin(epoch)
            epoch_logs, batch_logs = {}, {}
            for batch_index, batch in cur_iter:
                self.callbacks.on_batch_begin(batch_index, batch_logs)

                x, y = self.train_prepare_batch(batch)
                logits, reg_logits, loss, y = self.fit_handle(x=x, y=y, prefix='train_')

                batch_logs['loss'] = loss.item()

                batch_logs[self.train_monitor] = self.batch_metrics(logits, y)

                self.callbacks.on_batch_end(batch_index, batch_logs)

                # if batch_index == 5:
                #     break

            # Run on epoch end
            # 注意这个 epoch_logs 是共享变量, 在callbacks里面的类传递的!
            self.lr_handle(epoch, epoch_logs)
            self.callbacks.on_epoch_end(epoch, epoch_logs)

        self.callbacks.on_train_end()

        # Run on train end
        self.logger.info('Finished')
        if self.verbose:
            print('Finished.')

        if call_test:
            self.test()
