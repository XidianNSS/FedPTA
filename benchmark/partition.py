r"""
This file contains preset partitioners for the benchmarks.
All the Partitioner should implement the method `__call__(self, data)`
where `data` is the dataset to be partitioned and the return is a list of the partitioned result.

For example, The IIDPartitioner.__call__ receives a indexable object (i.e. instance of torchvision.datasets.mnsit.MNSIT)
and I.I.D. selects samples' indices in the original dataset as each client's local_movielens_recommendation data.
The list of sample indices are finally returnerd (e.g. [[0,1,2,...,1008], ...,[25,23,98,...,997]]).

To use the partitioner, you can specify Partitioner in the configuration dict for `flgo.gen_task`.
 Example 1: passing the parameter of __init__ of the Partitioner through the dict `para`

"""

from abc import abstractmethod, ABCMeta
import random
import numpy as np
import collections
from torch.utils.data import Dataset, DataLoader
import loguru
from copy import deepcopy
from typing import Union, List
import torch


try:
    import community.community_louvain
except:
    pass


class AbstractPartitioner(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class BasicPartitioner(AbstractPartitioner):
    """This is the basic class of data partitioner. The partitioner will be directly called by the
    task generator of different benchmarks. By overwriting __call__ method, different partitioners
    can be realized. The input of __call__ is usually a dataset.
    """
    def __call__(self, *args, **kwargs):
        return

    def register_generator(self, generator):
        r"""Register the generator as an self's attribute"""
        self.generator = generator

    def data_imbalance_generator(self, num_clients, datasize, imbalance=0):
        r"""
        Split the data size into several parts

        Args:
            num_clients (int): the number of clients
            datasize (int): the total data size
            imbalance (float): the degree of data imbalance across clients

        Returns:
            a list of integer numbers that represents local_movielens_recommendation data sizes
        """
        if imbalance == 0:
            samples_per_client = [int(datasize / num_clients) for _ in range(num_clients)]
            for _ in range(datasize % num_clients): samples_per_client[_] += 1
        else:
            imbalance = max(0.1, imbalance)
            sigma = imbalance
            mean_datasize = datasize / num_clients
            mu = np.log(mean_datasize) - sigma ** 2 / 2.0
            samples_per_client = np.random.lognormal(mu, sigma, (num_clients)).astype(int)
            thresold = int(imbalance ** 1.5 * (datasize - num_clients * 10))
            delta = int(0.1 * thresold)
            crt_data_size = sum(samples_per_client)
            # force current data size to match the total data size
            while crt_data_size != datasize:
                if crt_data_size - datasize >= thresold:
                    maxid = np.argmax(samples_per_client)
                    maxvol = samples_per_client[maxid]
                    new_samples = np.random.lognormal(mu, sigma, (10 * num_clients))
                    while min(new_samples) > maxvol:
                        new_samples = np.random.lognormal(mu, sigma, (10 * num_clients))
                    new_size_id = np.argmin(
                        [np.abs(crt_data_size - samples_per_client[maxid] + s - datasize) for s in new_samples])
                    samples_per_client[maxid] = new_samples[new_size_id]
                elif crt_data_size - datasize >= delta:
                    maxid = np.argmax(samples_per_client)
                    samples_per_client[maxid] -= delta
                elif crt_data_size - datasize > 0:
                    maxid = np.argmax(samples_per_client)
                    samples_per_client[maxid] -= (crt_data_size - datasize)
                elif datasize - crt_data_size >= thresold:
                    minid = np.argmin(samples_per_client)
                    minvol = samples_per_client[minid]
                    new_samples = np.random.lognormal(mu, sigma, (10 * num_clients))
                    while max(new_samples) < minvol:
                        new_samples = np.random.lognormal(mu, sigma, (10 * num_clients))
                    new_size_id = np.argmin(
                        [np.abs(crt_data_size - samples_per_client[minid] + s - datasize) for s in new_samples])
                    samples_per_client[minid] = new_samples[new_size_id]
                elif datasize - crt_data_size >= delta:
                    minid = np.argmin(samples_per_client)
                    samples_per_client[minid] += delta
                else:
                    minid = np.argmin(samples_per_client)
                    samples_per_client[minid] += (datasize - crt_data_size)
                crt_data_size = sum(samples_per_client)
        return samples_per_client


class IIDPartitioner(BasicPartitioner):
    """`Partition the indices of samples in the original dataset indentically and independently.

    Args:
        num_clients (int, optional): the number of clients
        imbalance (float, optional): the degree of imbalance of the amounts of different local_movielens_recommendation data (0<=imbalance<=1)
    """
    def __init__(self, num_clients: int = 100, imbalance: float = 0,
                 logger: loguru._Logger = loguru.logger):
        self.logger = logger
        self.num_clients = num_clients
        self.imbalance = imbalance

    def __str__(self):
        name = "IID Partitioner"
        if self.imbalance > 0:
            name += ', imbalance={:.2f}'.format(self.imbalance)
        return name

    def __call__(self, data):
        self.logger.info("IID partition started.")
        samples_per_client = self.data_imbalance_generator(self.num_clients, len(data), self.imbalance)
        d_idxs = np.random.permutation(len(data))
        local_datas = np.split(d_idxs, np.cumsum(samples_per_client))[:-1]
        local_datas = [di.tolist() for di in local_datas]

        self.logger.info("IID partition finished.")
        return local_datas


class DirichletPartitioner(BasicPartitioner):
    """`Partition the indices of samples in the original dataset according to Dirichlet distribution of the
    particular attribute. This way of partition is widely used by existing works in federated learning.

    Args:
        num_clients (int, optional): the number of clients
        alpha (float, optional): `alpha`(i.e. alpha>=0) in Dir(alpha*p) where p is the global distribution. The smaller alpha is, the higher heterogeneity the data is.
        imbalance (float, optional): the degree of imbalance of the amounts of different local_movielens_recommendation data (0<=imbalance<=1)
        error_bar (float, optional): the allowed error when the generated distribution mismatches the distirbution that is actually wanted, since there may be no solution for particular imbalance and alpha.
        index_func (func, optional): to index the distribution-dependent (i.e. label) attribute in each sample.
    """
    def __init__(self, num_clients: int = 100, alpha: float = 1.0, error_bar: float = 1e-6,
                 imbalance: float = 0, index_func=lambda x:x[-1],
                 logger: loguru._Logger = loguru.logger):
        self.logger = logger
        self.num_clients = num_clients
        self.alpha = alpha
        self.imbalance = imbalance
        self.index_func = index_func
        self.error_bar = error_bar

    def __str__(self):
        name = "Dirichlet partitioner, alpha={:.2f}, error_bar={}".format(self.alpha, self.error_bar)
        if self.imbalance > 0:
            name += ', imbalance={:.2f}'.format(self.imbalance)
        return name

    def __call__(self, data):
        self.logger.info("Dirichlet Partition started.")
        attrs = [self.index_func(d) for d in data]
        num_attrs = len(set(attrs))
        samples_per_client = self.data_imbalance_generator(self.num_clients, len(data), self.imbalance)
        # count the label distribution
        lb_counter = collections.Counter(attrs)
        lb_names = list(lb_counter.keys())
        p = np.array([1.0 * v / len(data) for v in lb_counter.values()])
        lb_dict = {}
        attrs = np.array(attrs)
        for lb in lb_names:
            lb_dict[lb] = np.where(attrs == lb)[0]
        proportions = [np.random.dirichlet(self.alpha * p) for _ in range(self.num_clients)]
        while np.any(np.isnan(proportions)):
            proportions = [np.random.dirichlet(self.alpha * p) for _ in range(self.num_clients)]
        sorted_cid_map = {k: i for k, i in zip(np.argsort(samples_per_client), [_ for _ in range(self.num_clients)])}
        error_increase_interval = 500
        max_error = self.error_bar
        loop_count = 0
        crt_id = 0
        crt_error = 100000
        while True:
            if loop_count >= error_increase_interval:
                loop_count = 0
                max_error = max_error * 10
            # generate dirichlet distribution till ||E(proportion) - P(D)||<=1e-5*self.num_classes
            mean_prop = np.sum([pi * di for pi, di in zip(proportions, samples_per_client)], axis=0)
            mean_prop = mean_prop / mean_prop.sum()
            error_norm = ((mean_prop - p) ** 2).sum()
            if crt_error - error_norm >= max_error:
                self.logger.debug("Dirichlet error: {:.8f}".format(error_norm))
                # print("Error: {:.8f}".format(error_norm))
                crt_error = error_norm
            if error_norm <= max_error:
                break
            excid = sorted_cid_map[crt_id]
            crt_id = (crt_id + 1) % self.num_clients
            sup_prop = [np.random.dirichlet(self.alpha * p) for _ in range(self.num_clients)]
            del_prop = np.sum([pi * di for pi, di in zip(proportions, samples_per_client)], axis=0)
            del_prop -= samples_per_client[excid] * proportions[excid]
            for i in range(error_increase_interval - loop_count):
                alter_norms = []
                for cid in range(self.num_clients):
                    if np.any(np.isnan(sup_prop[cid])):
                        continue
                    alter_prop = del_prop + samples_per_client[excid] * sup_prop[cid]
                    alter_prop = alter_prop / alter_prop.sum()
                    error_alter = ((alter_prop - p) ** 2).sum()
                    alter_norms.append(error_alter)
                if min(alter_norms) < error_norm:
                    break
            if len(alter_norms) > 0 and min(alter_norms) < error_norm:
                alcid = np.argmin(alter_norms)
                proportions[excid] = sup_prop[alcid]
            loop_count += 1
        local_datas = [[] for _ in range(self.num_clients)]
        self.dirichlet_dist = []  # for efficiently visualizing
        for lb in lb_names:
            lb_idxs = lb_dict[lb]
            lb_proportion = np.array([pi[lb_names.index(lb)] * si for pi, si in zip(proportions, samples_per_client)])
            lb_proportion = lb_proportion / lb_proportion.sum()
            lb_proportion = (np.cumsum(lb_proportion) * len(lb_idxs)).astype(int)[:-1]
            lb_datas = np.split(lb_idxs, lb_proportion)
            self.dirichlet_dist.append([len(lb_data) for lb_data in lb_datas])
            local_datas = [local_data + lb_data.tolist() for local_data, lb_data in zip(local_datas, lb_datas)]
        self.dirichlet_dist = np.array(self.dirichlet_dist).T
        for i in range(self.num_clients): np.random.shuffle(local_datas[i])
        self.local_datas = local_datas

        self.logger.info("Dirichlet partition finished.")
        return local_datas


class DiversityPartitioner(BasicPartitioner):
    """`Partition the indices of samples in the original dataset according to numbers of types of a particular
    attribute (e.g. label) . This way of partition is widely used by existing works in federated learning.

    Args:
        num_clients (int, optional): the number of clients
        diversity (float, optional): the ratio of locally owned types of the attributes (i.e. the actual number=diversity * total_num_of_types)
        imbalance (float, optional): the degree of imbalance of the amounts of different local_movielens_recommendation data (0<=imbalance<=1)
        index_func (int, optional): the index of the distribution-dependent (i.e. label) attribute in each sample.
    """
    def __init__(self, num_clients: int = 100, diversity: float = 1.0, index_func=lambda x:x[-1],
                 logger: loguru._Logger = loguru.logger):
        self.logger = logger
        self.num_clients = num_clients
        self.diversity = diversity
        self.index_func = index_func

    def __str__(self):
        name = "Diversity Partitioner, diversity={:.1f}".format(self.diversity)
        return name

    def __call__(self, data):
        self.logger.info("Diversity partition started.")
        labels = [self.index_func(d) for d in data]
        num_classes = len(set(labels))
        dpairs = [[did, lb] for did, lb in zip(list(range(len(data))), labels)]
        num = max(int(self.diversity * num_classes), 1)
        K = num_classes
        local_datas = [[] for _ in range(self.num_clients)]
        if num == K:
            for k in range(K):
                idx_k = [p[0] for p in dpairs if p[1] == k]
                np.random.shuffle(idx_k)
                split = np.array_split(idx_k, self.num_clients)
                for cid in range(self.num_clients):
                    local_datas[cid].extend(split[cid].tolist())
        else:
            times = [0 for _ in range(num_classes)]
            contain = []
            for i in range(self.num_clients):
                current = []
                j = 0
                while (j < num):
                    mintime = np.min(times)
                    ind = np.random.choice(np.where(times == mintime)[0])
                    if (ind not in current):
                        j = j + 1
                        current.append(ind)
                        times[ind] += 1
                contain.append(current)
            for k in range(K):
                idx_k = [p[0] for p in dpairs if p[1] == k]
                np.random.shuffle(idx_k)
                split = np.array_split(idx_k, times[k])
                ids = 0
                for cid in range(self.num_clients):
                    if k in contain[cid]:
                        local_datas[cid].extend(split[ids].tolist())
                        ids += 1

        self.logger.info("Diversity partition finished.")
        return local_datas


class GaussianPerturbationPartitioner(BasicPartitioner):
    """`Partition the indices of samples I.I.D. and bind additional and static gaussian noise to each sample, which is
    a setting of feature skew in federated learning.

    Args:
        num_clients (int, optional): the number of clients
        imbalance (float, optional): the degree of imbalance of the amounts of different local_movielens_recommendation data (0<=imbalance<=1)
        sigma (float, optional): the degree of feature skew
        scale (float, optional): the standard deviation of noise
        index_func (int, optional): the index of the feature to be processed for each sample.
    """
    def __init__(self, num_clients: int = 100, imbalance: float = 0.0, sigma: float = 0.1, scale: float = 0.1,
                 index_func=lambda x:x[0],
                 logger: loguru._Logger = loguru.logger):
        self.logger = logger
        self.num_clients = num_clients
        self.imbalance = imbalance
        self.sigma = sigma
        self.scale = scale
        self.index_func = index_func

    def __str__(self):
        name = "Gaussian Perturbation Partitioner, sigma={:.2f}, scale={:.2f}".format(self.sigma, self.scale)
        if self.imbalance > 0:
            name += ', imbalance={:.1f}'.format(self.imbalance)
        return name

    def __call__(self, data):
        self.logger.info("Gaussian perturbation partition started.")
        shape = tuple(np.array(self.index_func(data[0]).shape))
        samples_per_client = self.data_imbalance_generator(self.num_clients, len(data), self.imbalance)
        d_idxs = np.random.permutation(len(data))
        local_datas = np.split(d_idxs, np.cumsum(samples_per_client))[:-1]
        local_datas = [di.tolist() for di in local_datas]
        local_perturbation_means = [np.random.normal(0, self.sigma, shape) for _ in range(self.num_clients)]
        local_perturbation_stds = [self.scale * np.ones(shape) for _ in range(self.num_clients)]
        local_perturbation = []
        for cid in range(self.num_clients):
            c_perturbation = [np.random.normal(local_perturbation_means[cid], local_perturbation_stds[cid]).tolist() for
                              _ in range(len(local_datas[cid]))]
            local_perturbation.append(c_perturbation)
        self.local_perturbation = local_perturbation

        self.logger.info("Gaussian perturbation partition finished.")
        return local_datas


class IDPartitioner(BasicPartitioner):
    """`Partition the indices of samples I.I.D. according to the ID of each sample, which requires the passed parameter
    `data` has attribution `id`.

    Args:
        num_clients (int, optional): the number of clients
        priority (str, optional): The value should be in set ('random', 'max', 'min'). If the number of clients is smaller than the total number of all the clients, this term will decide the selected clients according to their local_movielens_recommendation data sizes.
    """
    def __init__(self, num_clients: int = -1, priority: str = 'random', index_func=lambda x:x.id,
                 logger: loguru._Logger = loguru.logger):
        self.logger = logger
        self.num_clients = int(num_clients)
        self.priorty = priority
        self.index_func = index_func

    def __str__(self):
        return 'ID Partitioner'

    def __call__(self, data):
        self.logger.info("ID partition started.")
        all_data = list(range(len(data)))
        data_owners = [self.index_func(d) for d in data]
        local_datas = collections.defaultdict(list)
        for idx in range(len(all_data)):
            local_datas[data_owners[idx]].append(idx)
        local_datas = list(local_datas.values())
        if self.num_clients < 0:
            self.num_clients = len(local_datas)
        elif self.priorty == 'max':
            local_datas = sorted(local_datas, key=lambda x: len('x'), reverse=True)[:self.num_clients]
        elif self.priorty == 'min':
            local_datas = sorted(local_datas, key=lambda x: len('x'))[:self.num_clients]
        elif self.priorty == 'random':
            random.shuffle(local_datas)
            local_datas = local_datas[:self.num_clients]
        local_datas = sorted(local_datas, key=lambda x:data[x[0]][self.index_func] if self.index_func is not None else data.id[x[0]])

        self.logger.info("ID partition finished.")
        return local_datas


class VerticalSplitPartitioner(BasicPartitioner):
    """`Partition the indices and shapes of samples in the original dataset for vertical federated learning. Different
    to the above partitioners, the partitioner.__call__ returns more flexible partition information instead of the indices that
    can be used to rebuild the partitioned data.

    Args:
        num_parties (int, optional): the number of parties
        imbalance (float, optional): the degree of imbalance of the number of features
        dim (int, optional): the dim of features to be partitioned
    """
    def __init__(self, num_parties: int = -1, imbalance: float = 0, dim: int = -1,
                 logger: loguru._Logger = loguru.logger):
        self.logger = logger
        self.num_parties = int(num_parties)
        self.imbalance = imbalance
        self.feature_pointers = []
        self.dim = dim

    def __str__(self):
        return 'Vertical Split Partitioner, imbalance{}'.format(self.imbalance)

    def __call__(self, data):
        self.logger.info("Vertical partition started.")
        local_datas = []
        feature = data[0][0]
        shape = feature.shape
        if self.dim == -1: self.dim = int(np.argmax(shape))
        self.num_parties = min(shape[self.dim], self.num_parties)
        feature_sizes = self.gen_feature_size(shape[self.dim], self.num_parties, self.imbalance)
        for pid in range(self.num_parties):
            pdata = {'sample_idxs': list(range(len(data))), 'pt_feature': (self.dim, feature_sizes, pid),
                     'with_label': (pid == 0)}
            local_datas.append(pdata)

        self.logger.info("Vertical partition finished.")
        return local_datas

    def gen_feature_size(self, total_size, num_parties, imbalance=0):
        size_partitions = []
        size_gen = self.integer_k_partition(total_size, num_parties)
        while True:
            try:
                tmp = next(size_gen)
                if tmp is not None:
                    size_partitions.append(tmp)
            except StopIteration:
                break
        size_partitions = sorted(size_partitions, key=lambda x: np.std(x))
        res = size_partitions[int(imbalance * (len(size_partitions) - 1))]
        return res

    def integer_k_partition(self, n, k, l=1):
        '''n is the integer to partition, k is the length of partitions, l is the min partition element size'''
        if k < 1:
            return None
        if k == 1:
            if n >= l:
                yield (n,)
            return None
        for i in range(l, n + 1):
            for result in self.integer_k_partition(n - i, k - 1, i):
                yield (i,) + result


class NodeLouvainPartitioner(BasicPartitioner):
    """
    Partition a graph into several subgraph by louvain algorithms. The input
    of this partitioner should be of type networkx.Graph
    """
    def __init__(self, num_clients: int = 100,
                 logger: loguru._Logger = loguru.logger):
        self.logger = logger
        self.num_clients = num_clients

    def __str__(self):
        name = "Node Louvain Partitioner"
        return name

    def __call__(self, data):
        self.logger.info("Node Louvain partition started.")
        local_nodes = [[] for _ in range(self.num_clients)]
        self.node_groups = community.community_louvain.best_partition(data)
        groups = collections.defaultdict(list)
        for ni, gi in self.node_groups.items():
            groups[gi].append(ni)
        groups = {k: groups[k] for k in list(range(len(groups)))}
        # ensure the number of groups is larger than the number of clients
        while len(groups) < self.num_clients:
            # find the group with the largest size
            groups_lens = [groups[k] for k in range(len(groups))]
            max_gi = np.argmax(groups_lens)
            # set the size of the new group
            min_glen = min(groups_lens)
            max_glen = max(groups_lens)
            if max_glen < 2 * min_glen: min_glen = max_glen // 2
            # split the group with the largest size into two groups
            nodes_in_gi = groups[max_gi]
            new_group_id = len(groups)
            groups[new_group_id] = nodes_in_gi[:min_glen]
            groups[max_gi] = nodes_in_gi[min_glen:]
        # allocate different groups to clients
        groups_lens = [groups[k] for k in range(len(groups))]
        group_ids = np.argsort(groups_lens)
        for gi in group_ids:
            cid = np.argmin([len(li) for li in local_nodes])
            local_nodes[cid].extend(groups[gi])

        self.logger.info("Node Louvain partition finished.")
        return local_nodes
#
# class KMeansPartitioner(BasicPartitioner):
#     """`Partition the indices of samples in the original dataset
#         according to their feature similarity.
#
#         Args:
#             num_clients (int, optional): the number of clients
#             imbalance (float, optional): the degree of imbalance of the amounts of different local_movielens_recommendation data (0<=imbalance<=1)
#         """
#
#     def __init__(self, num_clients=100, index_func=lambda x:x[0]):
#         self.num_clients = num_clients
#         self.index_func = index_func
#
#     def __str__(self):
#         name = ""
#         return name
#
#     def __call__(self, data):
#         features = [self.index_func(d) for d in data]
#         if len(features)==0: raise NotImplementedError("Feature indexed by {} was not found.".format(self.index_func))
#         if torch.is_tensor(features[0]):
#             features = np.array([t.view(-1).cpu().numpy() for t in features])
#         else:
#             features = np.array(features)
#         if len(features[0].shape)>1:
#             features = np.array([fe.flatten() for fe in features])
#         from scipy.cluster.vq import kmeans
#         centers = kmeans(features, self.num_clients)
#         local_datas = [[] for _ in self.num_clients]
#         for did, fe in enumerate(features):
#             cid = int((centers@fe).argmax())
#             local_datas[cid].append(did)
#         # local_datas = [di.tolist() for di in local_datas]
#         return local_datas


class MyDataset(Dataset):
    def __init__(self,
                 data: Union[List[float], List[int], np.ndarray, torch.Tensor],
                 targets: Union[List[float], List[int], np.ndarray, torch.Tensor] = None,
                 name: str = 'No Name Dataset'):
        """

        :param data: Union[List[float], List[int], np.ndarray, torch.Tensor]  x in dataset
        :param targets:  None if you want to create a dataset with no label
        :param name:
        """
        self.name = name

        if isinstance(data, torch.Tensor):
            self.data = data.to('cpu').detach().numpy()
        elif data is None:
            raise TypeError
        else:
            self.data = deepcopy(np.array(data))

        if isinstance(targets, torch.Tensor):
            self.targets = targets.to('cpu').detach().numpy()
            assert len(self.data) == len(self.targets)  # length check
        elif targets is None:
            self.targets = None
        else:
            self.targets = deepcopy(np.array(targets))
            assert len(self.data) == len(self.targets)  # length check

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """

        :param index:
        :return:  x, y if labeled dataset
                  x, None if not labeled
        """
        if self.targets is not None:
            x = self.data[index]
            y = self.targets[index]
            return x, y

        else:
            x = self.data[index]
            return x, None

    def labeled(self):
        """

        :return: bool, Ture if labeled, else False
        """
        if self.targets is None:
            return False
        else:
            return True

    def __str__(self):
        return self.name


class BasicDataLoaderGenerator(object):
    def __init__(self, dataset: MyDataset, partitioner: BasicPartitioner, batch_size=32,
                 logger: loguru._Logger = loguru.logger):
        """
        Accept a dataset and a partitioner, and will be able to create dataloaders.

        :param dataset: Private dataset.
        :param partitioner: Predefined partitioner
        :param batch_size:
        :param logger:
        """
        self.logger = logger
        self.dataset = deepcopy(dataset)
        self.partitioner = partitioner
        self.dataloaders = []
        self.batch_size = batch_size

        if hasattr(self.dataset, 'name'):
            self.dataset_name = self.dataset.name
        else:
            self.dataset_name = self.dataset.__class__.__name__
        self._initialize()
        self.logger.info('Data loaders generated.')

    def _initialize(self):
        if self.dataset.labeled():
            data_idx = self.partitioner(self.dataset)

            self.dataset_x = np.array(self.dataset.data)
            self.dataset_y = np.array(self.dataset.targets)
            for indices in data_idx:
                x = self.dataset_x[indices]
                y = self.dataset_y[indices]
                child_dataset = MyDataset(x, y, name=self.dataset_name)
                if len(child_dataset) <= 0:
                    self.logger.warning('Partition bug. Child dataset with length==0 encountered.')
                    while len(child_dataset) <= 0:
                        temp_indices = random.choice(data_idx)
                        x = self.dataset_x[temp_indices]
                        y = self.dataset_y[temp_indices]
                        child_dataset = MyDataset(x, y, name=self.dataset_name)

                self.dataloaders.append(DataLoader(child_dataset, batch_size=self.batch_size, shuffle=True))
        else:
            self.logger.error('You can not do partition on a dataset without label!')
            raise TypeError

    def __getitem__(self, item: int):
        return self.dataloaders[item]

    def __iter__(self):
        return self.dataloaders

