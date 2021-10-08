from tqdm import tqdm
import numpy as np

import torch
from collections import OrderedDict, Iterable
from models.utils import update_add_dicts, divide_dict, mkdir, Timer, pprint, FastAverager, FastAveragerDict, compute_confidence_interval
from typing import List, Iterable, Callable, Tuple
import warnings
import os
import csv
import io


# init 函数传入的 callbacks 是一些对象(操纵类)集合.
# CallbackList 控制 callbacks 之前加入的所有对象(以list形式).
# 在 fit 函数里, 调用 on_epoch_begin 一次 => \
#   调用callbacks里面所有类的on_epoch_begin函数. (callbacks里面类都是Callback的子类, on_epoch_begin是虚函数, 这样继承的)

# 请看每个 /experiments/ 下的(特定Approach)文件, \
#   里面callbacks(list)所包含的类是大家公有的, 但是每个类初始化的函数(类内所使用的函数)是不同的(因方法而变的), \
#   这就在few_shot文件夹下了.

class CallbackList(object):
    """Container abstracting a list of callbacks.

    # Arguments
        callbacks: List of `Callback` instances.
    """
    def __init__(self, callbacks):
        self.callbacks = [c for c in callbacks]

    def set_model_and_logger(self, model, logger):
        for callback in self.callbacks:
            # 每个对象设置到当前model.
            # 真正要看是怎么set_model的, 包括下面的 callback.操作函数 , 都要进到具体的对象里面看.

            callback.set_model_and_logger(model, logger)

    def on_epoch_begin(self, epoch, logs=None):
        """Called at the start of an epoch.
        # Arguments
            epoch: integer, index of epoch.
            logs: dictionary of logs.
        """
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch.
        # Arguments
            epoch: integer, index of epoch.
            logs: dictionary of logs.
        """
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch, logs=None):
        """Called right before processing a batch.
        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dictionary of logs.
        """

        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        """Called at the end of a batch.
        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dictionary of logs.
        """
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)

    def on_train_begin(self, logs=None):
        """Called at the beginning of training.
        # Arguments
            logs: dictionary of logs.
        """
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        """Called at the end of training.
        # Arguments
            logs: dictionary of logs.
        """
        for callback in self.callbacks:
            callback.on_train_end(logs)


# 父类共享方法框架. 但注意这里只是方法, 不存模型.
class Callback(object):
    def __init__(self):
        self.model = None
        self.logger = None
    def set_model_and_logger(self, model, logger):
        self.model = model
        self.logger = logger
    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

class ProgressBarLogger(Callback):
    """TQDM progress bar that displays the running average of loss and other metrics."""
    def __init__(self, length, avg_monitor, epoch_verbose=True):
        super(ProgressBarLogger, self).__init__()
        import torchvision
        self.length = length
        self.epoch_verbose = epoch_verbose
        self.avg_monitor = avg_monitor
        self.avg_acc = FastAverager()

    def on_epoch_begin(self, epoch, logs=None):
        if self.epoch_verbose:
            self.pbar = tqdm(total=self.length, desc='Epoch {}'.format(epoch))
            # TensorBoard Test:
            # from torch.utils.tensorboard import SummaryWriter
            # self.writer = SummaryWriter('runs/epoch-' + str(epoch))
            self.seen = 0
            self.avg_acc.reset()

    def on_batch_end(self, batch, logs=None):
        if self.epoch_verbose:
            self.seen += 1
            self.pbar.update(1)
            self.pbar.set_postfix(logs)
            self.avg_acc.add(logs[self.avg_monitor])

    def on_epoch_end(self, epoch, logs=None):
        if self.epoch_verbose:
            self.pbar.close()
            self.logger.info('Avg train_ accuracy: %.4f' % self.avg_acc.item())
            print('Avg train_ accuracy: %.4f' % self.avg_acc.item())

# 未改, 最好在 eval_fn 调用处要改一下.
class EvaluateFewShot(Callback):
    """Evaluate a network on an k-shot, k-way classification tasks after every epoch.

    # Arguments
        eval_fn: Callable to perform few-shot classification. Examples include `proto_net_episode`,
            `matching_net_episode` and `meta_gradient_step` (MAML).
        num_tasks: int. Number of k-shot classification tasks to evaluate the model with.
        k_shot: int. Number of samples for each class in the k-shot classification tasks.
        k_way: int. Number of classes in the k-shot classification tasks.
        q_queries: int. Number query samples for each class in the k-shot classification tasks.
        task_loader: Instance of ShotWrapper class
        prepare_batch: function. The preprocessing function to apply to samples from the dataset.
        prefix: str. Prefix to identify dataset.
    """

    def __init__(self,
                 val_loader: torch.utils.data.DataLoader,
                 test_loader: torch.utils.data.DataLoader,
                 val_prepare_batch: Callable,
                 test_prepare_batch: Callable,
                 batch_metrics: Callable,
                 assist_metrics: Callable,
                 eval_fn: Callable,
                 test_interval: int,
                 max_epoch: int,
                 model_filepath: str,
                 monitor: str,
                 save_best_only: bool = True,
                 mode: str = 'max',
                 simulation_test: bool = False,
                 verbose: bool = True,
                 epoch_verbose: bool = True,
                 val_base_loader: torch.utils.data.DataLoader = None,
                 test_base_loader: torch.utils.data.DataLoader = None,
                 **kargs
                 ):
        super(EvaluateFewShot, self).__init__()
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.val_prepare_batch = val_prepare_batch
        self.test_prepare_batch = test_prepare_batch
        self.eval_fn = eval_fn
        self.simulation_test = simulation_test
        self.test_interval_step = 0
        self.test_interval = test_interval

        # model check point:
        self.max_epoch = max_epoch
        
        self.model_filepath = model_filepath
        if model_filepath is not None:
            mkdir(model_filepath[:model_filepath.rfind('/')])
        else:
            import warnings
            warnings.warn("The model file path is not set.",
                          UserWarning)
        # self.model_filepath_test_best = model_filepath_test_best
        # mkdir(model_filepath_test_best[:model_filepath_test_best.rfind('/')])
        self.batch_metrics = batch_metrics
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.verbose = verbose
        self.epoch_verbose = epoch_verbose
        self.kargs = kargs

        self.assist_metrics = assist_metrics
        if self.assist_metrics is not None:
            # 记录每个epoch的结果:
            self.assist_metrics_log = FastAveragerDict()

        if mode not in ['min', 'max']:
            raise ValueError('Mode must be one of (min, max).')

        self.val_best, self.test_best = None, None

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater

        self.val_base_loader, self.test_base_loader = val_base_loader, test_base_loader

    def on_train_begin(self, logs=None):
        if self.verbose:
            self.timer = Timer(self.max_epoch)

    def predict_log(self, epoch, prefix, logs=None):
        dataloader = eval(f'self.{prefix}loader')
        base_dataloader = eval(f'self.{prefix}base_loader')

        seen = 0
        loss_name = prefix + 'loss'
        if logs is None:
            logs = {}
        logs[loss_name], logs[self.monitor] = 0, 0

        if self.assist_metrics is not None:
            assist_metrics_acc = FastAveragerDict()

        with torch.no_grad():
            # 这里开始做epoch_end的val:
            # is_hc 与 is_gfsl 所构造的base_dataloader是不同的, 详见helper.py.
            if self.kargs['is_hc']:
                cur_iter = enumerate(tqdm(zip(dataloader, base_dataloader), total=len(dataloader)))
            elif self.kargs['is_gfsl']:
                cur_iter = enumerate(zip(dataloader, base_dataloader))
                if prefix == 'test_':
                    cur_iter_timer = Timer(len(dataloader))
            elif self.epoch_verbose:
                cur_iter = enumerate(tqdm(dataloader))
            else:
                cur_iter = enumerate(dataloader)

            metrics_results = []

            icassp = True if prefix == 'val_' else False
            if icassp:
                icassp_acc_log = []
            for batch_idx, batch in cur_iter:
                x, y = eval('self.' + prefix + 'prepare_batch')(batch)

                # 这里eval_fn就是该类(在callbacks(list))初始化时传进来的函数, 在/few_shot/下. \
                #   比如 matching_net_episode.

                # on_epoch_end 这里的: \
                #   这里看 诸如matching_net_episode 的传参是什么, 请注意传的model就是 set_model_and_logger 里面设置的model, \
                #   实际上就是! fit 函数时传进来的model. 这个model就是在models.py里面定义的.
                # 注意这里的传参逻辑一定要搞清楚.

                # 注意这里就是测试过程了呀, 完全在evaluation文件夹下数据上测, 注意train=False, 虽然还是用 proto_net_episode.
                # 就相当于forward.

                if self.assist_metrics is not None:
                    logits, reg_logits, loss, y, y_unseen = self.eval_fn(
                        x=x,
                        y=y,
                        prefix=prefix
                    )
                else:
                    logits, reg_logits, loss, y = self.eval_fn(
                        x=x,
                        y=y,
                        prefix=prefix
                    )

                seen += 1
                logs[loss_name] += loss.item()

                if prefix == 'test_':
                    metrics_results.append(self.batch_metrics(logits, y))
                # elif prefix == 'val_':
                else:
                    logs[self.monitor] += self.batch_metrics(logits, y)
                # print('test acc:', metrics_results[-1])
                # print(self.batch_metrics(logits, y), loss.item())

                if self.assist_metrics is not None:
                    cur_assist_metrics_dict = self.assist_metrics(logits, y, y_unseen)
                    assist_metrics_acc.add(cur_assist_metrics_dict)
                    # ICASSP 辅助实验: 在此往文件记录seen, unseen 和 harmonic_mean的val值. 后续给df读.
                    if icassp:
                        icassp_acc_log.append((cur_assist_metrics_dict['seen'], cur_assist_metrics_dict['unseen'], cur_assist_metrics_dict['harmonic_mean']))

                if self.kargs['is_gfsl'] and prefix == 'test_':
                    print('\rETA: {} / {}.        '.format(*cur_iter_timer.measure(batch_idx)), end="")


            if icassp:
                with open('/home/zhangyk/Few-shot-Framework/run/icassp_ablation_study.csv', 'a') as f:
                    for i in icassp_acc_log:
                        f.write(f'{epoch},{i[0]},{i[1]},{i[2]}\n')

            if self.assist_metrics is not None:
                assist_metrics_result = assist_metrics_acc.item()

                if prefix == 'val_':
                    self.assist_metrics_log.add(assist_metrics_result)

                assist_metrics_result_str = dict(sorted({k: '%.2f' % (v*100) for k, v in assist_metrics_result.items()}.items()))
                print(f'Acc per Class: {assist_metrics_result_str}')
                self.logger.info(assist_metrics_result_str)

            logs[loss_name] /= seen

            if prefix == 'test_':
                logs[self.monitor], logs[self.monitor + ' (conf)'] = compute_confidence_interval(metrics_results)

                if self.model.args.is_alchemy:
                    # 调参模式的统一记录文件.
                    from models.utils import alchemy_log
                    alchemy_log(f'{self.model.args.time_str}: ' + '%.4f, %.3f' % (logs[self.monitor] * 100, logs[self.monitor + ' (conf)'] * 100))
            # elif prefix == 'val_':
            else:
                if self.assist_metrics is not None:
                    logs[self.monitor] = assist_metrics_result['harmonic_mean']
                else:
                    logs[self.monitor] /= seen

            self.logger.info('Epoch: %d:' % epoch)
            for k, v in logs.items():
                self.logger.info(f'{k}: {v}')

            if self.verbose:
                print('Epoch: %d:' % epoch)
                pprint(logs)

            return logs[self.monitor]

    def on_epoch_end(self, epoch, logs=None):
        self.test_interval_step += 1
        # if self.rougee != None:
        #     self.rougee.predict_log(self.model, logs)

        self.predict_log(epoch, 'val_', logs)
        if self.simulation_test or self.test_interval_step == self.test_interval:
            self.predict_log(epoch, 'test_', logs)
            self.test_interval_step = 0

        if self.judge_monitor(logs):
            # if self.verbose > 0:
            #     # print('\nEpoch %d: saving model to [%s].' % (epoch + 1, self.model_filepath))
            #     self.logger.info('Saving model.')
            self.val_best = logs.get(self.monitor)
            if self.model_filepath is not None:
                torch.save(self.model.state_dict(), self.model_filepath)

            val_acc_str = 'Best val_ accuracy: {:.4f}.'.format(self.val_best)

            self.logger.info('Saving val_best model.')

            self.logger.info(val_acc_str)
            if self.verbose:
                print(val_acc_str)

            # # save 完model直接测一次.
            # self.predict_log(epoch, 'test_', logs)
            # # 判断是否需要更新测试集上最好结果:
            # test_current = logs.get(self.test_monitor)
            # if self.test_best == None or self.monitor_op(test_current, self.test_best):
            #     self.test_best = test_current
            #     torch.save(self.model.state_dict(), self.model_filepath_test_best)
            #     if self.verbose > 0:
            #         self.logger.info('Saving test_best model.')

        eta_str = 'ETA: {} / {}.'.format(*self.timer.measure(epoch))
        self.logger.info(eta_str)
        self.logger.info('')
        if self.verbose:
            print(eta_str)
            print()

    # TODO: 期望logs是记录了所有结果的.

    def judge_monitor(self, logs):
        """判断当前验证集上是否表现更好:
    
        # Argument
            logs: 验证集当前最好准确率.
        # Return
            bool: 是否需要保存模型.
        """
        if self.val_best == None:
            return True
        val_current = logs.get(self.monitor)

        return self.monitor_op(val_current, self.val_best)
        # if self.test_monitor in logs.keys():
        #     test_current = logs.get(self.test_monitor)
        #     if self.test_best != None and self.monitor_op(test_current, self.test_best):
        #         if not self.monitor_op(val_current, self.val_best):
        #             warnings.warn('测试更好, 但是验证更菜.', RuntimeWarning)
        #             self.logger.warning('测试更好, 但是验证更菜.')
        #         return True
        # else:
        #     return self.monitor_op(val_current, self.val_best)



# class ModelCheckpoint(Callback):
#     """Save the model after every epoch.

#     `model_filepath` can contain named formatting options, which will be filled the value of `epoch` and keys in `logs`
#     (passed in `on_epoch_end`).

#     For example: if `model_filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`, then the model checkpoints will be saved
#     with the epoch number and the validation loss in the filename.

#     # Arguments
#         model_filepath: string, path to save the model file.
#         monitor: quantity to monitor.
#         verbose: verbosity mode, 0 or 1.
#         save_best_only: if `save_best_only=True`,
#             the latest best model according to
#             the quantity monitored will not be overwritten.
#         mode: one of {auto, min, max}.
#             If `save_best_only=True`, the decision
#             to overwrite the current save file is made
#             based on either the maximization or the
#             minimization of the monitored quantity. For `val_acc`,
#             this should be `max`, for `val_loss` this should
#             be `min`, etc. In `auto` mode, the direction is
#             automatically inferred from the name of the monitored quantity.
#         save_weights_only: if True, then only the model's weights will be
#             saved (`model.save_weights(model_filepath)`), else the full model
#             is saved (`model.save(model_filepath)`).
#         period: Interval (number of epochs) between checkpoints.
#     """

#     def __init__(self, model_filepath, monitor, save_best_only=True, mode='max', verbose=True):
#         super(ModelCheckpoint, self).__init__()
#         self.model_filepath = model_filepath
#         mkdir(model_filepath[:model_filepath.rfind('/')])
#         self.val_monitor = 'val_' + monitor
#         self.test_monitor = 'test_' + monitor
#         self.save_best_only = save_best_only
#         self.verbose = verbose

#         if mode not in ['min', 'max']:
#             raise ValueError('Mode must be one of (min, max).')

#         self.val_best, self.test_best = None, None

#         if mode == 'min':
#             self.monitor_op = np.less
#         elif mode == 'max':
#             self.monitor_op = np.greater
    
#     def judge_monitor(self, logs):
#         if self.val_best == None or self.test_best == None:
#             return True

#         val_current = logs.get(self.val_monitor)
#         if self.test_monitor in logs.keys():
#             test_current = logs.get(self.test_monitor)
#             if self.monitor_op(test_current, self.test_best):
#                 if not self.monitor_op(val_current, self.val_best):
#                     warnings.warn('测试更好, 但是验证更菜.', RuntimeWarning)
#                     self.logger.warning('测试更好, 但是验证更菜.')
#                 return True
#         else:
#             return self.monitor_op(val_current, self.val_best)

        
#     def on_epoch_end(self, epoch, logs=None):
#         # TODO: 这里的model_filepath没有嵌入epoch.
#         # model_filepath = self.model_filepath.format(epoch=epoch + 1, **logs)
#         # 为了保证传进来__init__的model_filepath就是最优模型的, 这里不改变model_filepath.

#         if self.judge_monitor(logs):
#             if self.verbose > 0:
#                 # print('\nEpoch %d: saving model to [%s].' % (epoch + 1, self.model_filepath))
#                 self.logger.info('Saving model.')
#             self.val_best, self.test_best = logs.get(self.val_monitor), logs.get(self.test_monitor)
#             torch.save(self.model.state_dict(), self.model_filepath)

class CSVLogger(Callback):
    """Callback that streams epoch results to a csv file.
    Supports all values that can be represented as a string,
    including 1D iterables such as np.ndarray.

    # Arguments
        filename: filename of the csv file, e.g. 'run/log.csv'.
        separator: string used to separate elements in the csv file.
        append: True: append if file exists (useful for continuing
            training). False: overwrite existing file,
    """

    def __init__(self, filename, separator=',', append=False):
        self.sep = separator
        self.filename = filename
        mkdir(filename[:filename.rfind('/')])
        self.append = append
        self.writer = None
        self.append_header = True
        self.file_flags = ''
        self._open_args = {'newline': '\n'}
        super(CSVLogger, self).__init__()

        if self.append:
            if os.path.exists(self.filename):
                with open(self.filename, 'r' + self.file_flags) as f:
                    self.append_header = not bool(len(f.readline()))
            mode = 'a'
        else:
            mode = 'w'

        self.csv_file = io.open(self.filename,
                                mode + self.file_flags,
                                **self._open_args)

    # 在这里将各种参数写入文件.
    def on_epoch_end(self, epoch, logs=None):
        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, str):
                return k
            elif isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k

        if not self.writer:
            class CustomDialect(csv.excel):
                delimiter = self.sep
            fieldnames = ['epoch'] + list(logs.keys())
            self.writer = csv.DictWriter(self.csv_file,
                                         fieldnames=fieldnames,
                                         dialect=CustomDialect)
            if self.append_header:
                self.writer.writeheader()

        row_dict = OrderedDict({'epoch': epoch})
        row_dict.update((key, handle_value(logs[key])) for key in logs.keys())
        # row_dict 就是 csv 文件里记录的信息.
        self.writer.writerow(row_dict)
        self.csv_file.flush()

    def on_train_end(self, logs=None):
        self.csv_file.close()
        self.writer = None
