import os
import logging
from collections import OrderedDict
from torchstat import stat

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from thop import profile
import models.networks as networks
from .base_model import BaseModel
#torch.set_default_tensor_type(torch.DoubleTensor)
logger = logging.getLogger('base')
#torch.set_default_tensor_type(torch.DoubleTensor)
#torch.set_default_tensor_type(torch.FloatTensor)
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
class forceModel(BaseModel):
    def __init__(self, opt):
        super(forceModel, self).__init__(opt)
        train_opt = opt['train']

        # define network and load pretrained models
       # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        device = torch.device("cuda:1" )
        self.netG = networks.define_G(opt).to(self.device)
        self.load()
        print('cuda')

        if self.is_train:
            self.netG.train()

            # loss
            loss_type = train_opt['criterion']
            if loss_type == 'l1':
                self.cri = nn.L1Loss().to(self.device)
            elif loss_type == 'l2':
                self.cri = nn.MSELoss().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
            self.l_w = train_opt['weight']

            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            momentum=train_opt['momentum']
            optim_params = []
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    logger.warning('Params [{:s}] will not optimize.'.format(k))
         #   self.optimizer_G = torch.optim.SGD(optim_params, lr=train_opt['lr_G'], momentum=momentum)            
            self.optimizer_G = torch.optim.Adam(
                optim_params, lr=train_opt['lr_G'], weight_decay=wd_G)
            self.optimizers.append(self.optimizer_G)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(lr_scheduler.MultiStepLR(optimizer, \
                        train_opt['lr_steps'], train_opt['lr_gamma']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()
        # print network
        self.print_network()

    def feed_data(self, data):
        self.img = data['img'].to(self.device)  # img
        self.real = data['force'].to(self.device)  # fz
       # self.real_xs = data['forcexs'].to(self.device)  # xs
      #  self.real_ys = data['forceys'].to(self.device)  # ys

    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()
        self.fake = self.netG(self.img)
      #  print("fake.shape")
    #    print(self.fake.shape)
   #     self.fake_fz=self.fake_fz.float()
        l = self.l_w * self.cri(self.fake[:,0], self.real[:,0])+self.l_w * self.cri(self.fake[:,1], self.real[:,1])+self.l_w * self.cri(self.fake[:,2], self.real[:,2])      
       # print(self.fake_fz)
     #   l.double()
     #   print("============")
        l.backward()
        self.optimizer_G.step()

        # set log
        self.log_dict['loss'] = l.item()

    def test(self):
        self.netG.eval()
    #    print(self.img.shape)
      #  logger.info(stat(self.netG,(3,819,819)))
     #   flops, params = profile(self.netG, inputs=(self.img, ))
      #  logger.info('FLOPs = ' + str(flops/1000**3) + 'G')
       # logger.info('Params = ' + str(params/1000**2) + 'M')
        with torch.no_grad():
            self.fake = self.netG(self.img)
        self.netG.train()


    def get_current_log(self):
        return self.log_dict
        
    def get_current_force(self):
        f_f = self.fake.detach().float().cpu()
        r_f = self.real.detach().float().cpu()
        return {'f_f': f_f, 'r_f': r_f}

    def get_current_fz(self):
        f_fz = self.fake_fz.detach()[0].float().cpu()
        r_fz = self.real_fz.detach()[0].float().cpu()
        return {'f_fz': f_fz, 'r_fz': r_fz}

    def get_current_visuals(self, need_HR=True):
        out_dict = OrderedDict()
        out_dict['LR'] = self.var_L.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        if need_HR:
            out_dict['HR'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading pretrained model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG)
            #print(self.netG)

    def save(self, iter_step):
        self.save_network(self.netG, 'G', iter_step)
