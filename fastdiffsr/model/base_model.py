import os
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel


class BaseModel():
    def __init__(self, opt):
        self.opt = opt
        if opt['gpu_ids'] is not None and torch.cuda.is_available():
            if opt.get('dist'):
                self.device = torch.device('cuda', opt['local_rank'])
            else:
                self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.begin_step = 0
        self.begin_epoch = 0

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        pass

    def get_current_losses(self):
        pass

    def print_network(self):
        pass

    def set_device(self, x):
        if isinstance(x, dict):
            for key, item in x.items():
                if item is not None:
                    x[key] = item.to(self.device)
        elif isinstance(x, list):
            for idx, item in enumerate(x):
                if item is not None:
                    x[idx] = item.to(self.device)
        else:
            x = x.to(self.device)
        return x

    def wrap_network(self, network):
        if self.opt.get('dist'):
            return DistributedDataParallel(
                network,
                device_ids=[self.opt['local_rank']],
                output_device=self.opt['local_rank'],
                find_unused_parameters=True)
        if self.opt['gpu_ids'] and len(self.opt['gpu_ids']) > 1:
            return nn.DataParallel(network)
        return network

    def get_bare_model(self, network):
        if isinstance(network, (nn.DataParallel, DistributedDataParallel)):
            network = network.module
        return network

    def get_network_description(self, network):
        '''Get the string and total parameters of the network'''
        network = self.get_bare_model(network)
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n

    def getFlopsAndParams(self, network):
        from thop import clever_format, profile

        network = self.get_bare_model(network)
        input = torch.randn(1, 3, 256, 256)
        total_ops, total_params = profile(network, inputs=(input,))
        total_ops, total_params = clever_format([total_ops, total_params], '%.3f')
        return [total_ops, total_params]
