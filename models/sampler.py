from typing import List, Iterable
import numpy as np

import torch
from torch.utils.data import Sampler
from torch.distributions.multivariate_normal import MultivariateNormal

# 该文件中 传入Dataloader的Sampler类函数 是数据集特异的, 所以**作为数据集类的成员对象**.

class ShotTaskSamplerForList(Sampler):
    def __init__(self, dataset, episodes_per_epoch, shot, way, query, num_tasks):
        # 要求dataset.label一定是完整有序的, 下标即与数据点对应的.
        super(ShotTaskSamplerForList, self).__init__(dataset)
        self.episodes_per_epoch = episodes_per_epoch
        self.way = way
        self.shot = shot
        self.n_per = shot + query
        self.num_tasks = num_tasks

        label = np.array(dataset.label)
        self.m_idx = []
        for i in set(dataset.label):
            idx = np.argwhere(label == i).reshape(-1)
            idx = torch.from_numpy(idx)
            self.m_idx.append(idx)

    def __len__(self):
        return self.episodes_per_epoch

    def __iter__(self):
        for i_batch in range(self.episodes_per_epoch):
            # sample 出way个class:
            batch = []
            classes = torch.randperm(len(self.m_idx))[:self.way]
            for task in range(self.num_tasks):
                for c in classes:
                    l = self.m_idx[c]
                    assert len(l) >= self.n_per, '检查shot数是否大于类别下的样本总数.'
                    pos = torch.randperm(len(l))[:self.n_per]
                    batch.append(l[pos])

            batch = torch.cat(batch)
            # 转换为support在前面的形式:
            batch = torch.cat([
                torch.cat([batch[(i*self.n_per) : (self.shot+i*self.n_per)] for i in range(self.way)]),
                torch.cat([batch[(self.shot+i*self.n_per) : ((i+1)*self.n_per)] for i in range(self.way)])
                ])

            yield batch

# 因为Dataloader传入的数据集是 map-style datasets, (这也仅在这种下标数据集上使用.)
# 所以下面传入的ShotTaskSampler(batch_sampler参数) yields a list of batch idxices.
class ShotTaskSamplerForDataFrame(Sampler):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        episodes_per_epoch: int,
        num_tasks: int,
        shot: int,
        way: int,
        query: int
        ):
        """PyTorch Sampler subclass that generates batches of shot-shot, way-way, query-query tasks.

        Each shot-shot task contains a "support set" of `way` sets of `shot` samples and a "query set" of `way` sets
        of `query` samples. The support set and the query set are all grouped into one Tensor such that the first shot * way
        samples are from the support set while the remaining query * way samples are from the query set.

        分 shot*way 和 query*way, 并且是不相交的.

        The support and query sets are sampled such that they are disjoint i.e. do not contain overlapping samples.

        请注意下面 num_tasks 和 episodes_per_epoch 的区别.
        # Arguments
            dataset: Instance of torch.utils.data.Dataset from which to draw samples
            episodes_per_epoch: Arbitrary number of batches of shot-shot tasks to generate in one epoch
            k_shot: int. Number of samples for each class in the shot-shot classification tasks.
            k_way: int. Number of classes in the shot-shot classification tasks.
            q_queries: int. Number query samples for each class in the shot-shot classification tasks.
            num_tasks: Number of shot-shot tasks to group into a single batch
            fixed_tasks: If this argument is specified this Sampler will always generate tasks from
                the specified classes
        """
        super(ShotTaskSamplerForDataFrame, self).__init__(dataset)
        self.episodes_per_epoch = episodes_per_epoch
        self.dataset = dataset

        if num_tasks < 1:
            raise ValueError('num_tasks must be > 1.')

        self.num_tasks = num_tasks
        # TODO: Raise errors if initialise badly
        self.way = way
        self.shot = shot
        self.query = query

        self.i_task = 0

    def __len__(self):
        return self.episodes_per_epoch

    def __iter__(self):
        for _ in range(self.episodes_per_epoch):
            batch = []

            # 每个 episode 一切都重新来了.
            for task in range(self.num_tasks):
                # Get random classes
                # 随机sample way 个类:
                episode_classes = np.random.choice(self.dataset.df['class_id'].unique(),
                                                    size=min(self.way, len(self.dataset.df['class_id'].unique())),
                                                    replace=False)
                # print("fixed_tasks is None, episode_classes:", episode_classes)

                df = self.dataset.df[self.dataset.df['class_id'].isin(episode_classes)]

                # TODO: support和query不要分开来sample.
                # support_k 字典, <key: way 个class, value: 每个class对应的n个数据.
                support_k = {way: None for way in episode_classes}
                for way in episode_classes:
                    # Select support examples
                    # 随机sample shot 个样本, 在之前sample的k类的每类下.
                    support = df[df['class_id'] == way].sample(n=self.shot)
                    support_k[way] = support

                    for i, s in support.iterrows():
                        batch.append(s['id'])

                for way in episode_classes:
                    # 随机sample query 个样本, 在之前sample的k类的每类下, 但是需要保证不和support有交集.
                    query = df[(df['class_id'] == way) & (~df['id'].isin(support_k[way]['id']))].sample(n=self.query)
                    for i, query in query.iterrows():
                        batch.append(query['id'])

            # 请特别注意, batch里面的是按照顺序排的:
            #   shot 个support 出现 way 次(因为k个不同类) + query 个query 出现 way 次.
            #   也就是 (shot * way + query * way), 这样构成了一个task, 这可以有很多个task, 请注意这里需要设置几个task, ProtoNet这里是一个.
            #   也就是 (shot) 个这样一组是 一类, 达到 (shot * way) 之后 (query) 个这样一组是一类.
            # episodes_per_epoch 决定了有多少个这样的多task训练.

            yield np.stack(batch)


class RandomSampler():
    def __init__(self, label, n_batch, n_per):
        self.n_batch = n_batch
        self.n_per = n_per
        self.label = np.array(label)
        self.num_label = self.label.shape[0]

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = torch.randperm(self.num_label)[:self.n_per]
            yield batch


class MixedMultivariateNormal():
    def __init__(self, base_means, base_covs):
        self.base_means = base_means.cpu().detach().numpy()
        self.base_covs = base_covs + (torch.eye(base_covs.shape[1]).to(torch.device('cuda')) * 1e-5).repeat(base_covs.shape[0], 1, 1)
        self.base_covs = self.base_covs.cpu().detach().numpy()
        self.MIXED_WEIGHTS = [0.75, 0.5] # 决定每个类sample的数量. 这里取定 top_k = 2.
        # self.base_sampled = [MultivariateNormal(m, c + torch.eye(c.shape[0]).to(torch.device('cuda')) * 1e-5) for m, c in zip(base_means, base_covs)]

    # 混合多元高斯的sample, 根据选定的class id来生成混合高斯.
    def sample(self, support_mean, num_sampled, topk_idxex=[]):
        # assert len(topk_idxex) == 2
        support_cov = np.mean(self.base_covs[topk_idxex], axis=0)
        
        num_sampled_list = [int(num_sampled * i) for i in self.MIXED_WEIGHTS]
        # 在refer_classes的高斯中采样:
        classes_sampled = [np.random.multivariate_normal(self.base_means[i], self.base_covs[i], n) for i, n in zip(topk_idxex, num_sampled_list)]
        classes_sampled.insert(0, np.random.multivariate_normal(support_mean, support_cov, num_sampled))
        classes_sampled = np.concatenate(classes_sampled)
        from sklearn.mixture import GaussianMixture
        gm = GaussianMixture(n_components=3).fit(classes_sampled)
        gm_sampled, _ = gm.sample(num_sampled)
        return gm_sampled