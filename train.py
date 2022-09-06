import os.path
import sys
import math
import argparse
import time
import random
import numpy as np
from collections import OrderedDict
import logging
import time
import torch
import math
import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model
import os
import torch.nn as nn
from data.util import bgr2ycbcr
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#torch.set_default_tensor_type(torch.DoubleTensor)
def main():
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option JSON file.')
    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt = option.dict_to_nonedict(opt)  # Convert to NoneDict, which return None for missing key.

    # train from scratch OR resume training
    if opt['path']['resume_state']:  # resuming training
        resume_state = torch.load(opt['path']['resume_state'])
    else:  # training from scratch
        resume_state = None
        util.mkdir_and_rename(opt['path']['experiments_root'])  # rename old folder if exists
        util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                     and 'pretrain_model' not in key and 'resume' not in key))

    # config loggers. Before it, the log will not work
    util.setup_logger(None, opt['path']['log'], 'train', level=logging.INFO, screen=True)
    util.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')

    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))
        option.check_resume(opt)  # check resume options

    logger.info(option.dict2str(opt))
    # tensorboard logger
    if opt['use_tb_logger'] and 'debug' not in opt['name']:
        from tensorboardX import SummaryWriter
        tb_logger = SummaryWriter(log_dir='../tb_logger/' + opt['name'])

    # random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benckmark = True
    # torch.backends.cudnn.deterministic = True

    # create train and val dataloader
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                len(train_set), train_size))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                total_epochs, total_iters))
            train_loader = create_dataloader(train_set, dataset_opt)
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt)
            logger.info('Number of val images in [{:s}]: {:d}'.format(dataset_opt['name'],
                                                                      len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None
    print("data success") 
    # create model
    model = create_model(opt)
    print("model success") 
    # resume training
    if resume_state:
        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    # training
  #  eng = matlab.engine.connect_matlab()
    cri=nn.MSELoss()
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    for epoch in range(start_epoch, total_epochs):
        for _, train_data in enumerate(train_loader):
        #    print("=============")
          #  print(len(train_data))
            current_step += 1
            if current_step > total_iters:
                break
            # update learning rate
            time_ul=time.time()
            model.update_learning_rate()
            time_ul=time.time()-time_ul
          #  print("update learning rate")
          #  print(time_ul)
          #  print("============")

            # training
            time_tra=time.time()
            model.feed_data(train_data)
        #    print('----')
        #    print(train_data)
        #    print(train_data.shape)
       #     print("------")
            model.optimize_parameters(current_step)
            time_tra=time.time()-time_tra
         #   print("training")
          #  print(time_tra)
          #  print(time.time())
           # print("============")

            # log
            time_log=time.time()
            if current_step % opt['logger']['print_freq'] == 0:
                logs = model.get_current_log()
                fz = model.get_current_force()
                f_f = fz['f_f']
                r_f = fz['r_f']
              #  logger.info(f"fake is {f_f}")
              #  logger.info(f"real is {r_f}")
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
                    epoch, current_step, model.get_current_learning_rate())
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    # tensorboard logger
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        tb_logger.add_scalar(k, v, current_step)
                logger.info(message)
            time_log=time.time()-time_log
           # print("log")
           # print(time_log)
          #  print("============")

            # validation
            if current_step % opt['train']['val_freq'] == 0:
                idx = 0
                fx_rmse =0.0
                fy_rmse =0.0
                fz_rmse =0.0
                los=0.0
                
              #  time_vla=time.time()
                for val_data in val_loader:
                    idx += 1
                    model.feed_data(val_data)
                    model.test()
                    fz = model.get_current_force()
                    f_f=fz['f_f']
                    r_f=fz['r_f']
                  #  logger.info(f"fake is {f_f}")
                  #  logger.info(f"real is {r_f}")
                    x_rmse= util.calculate_mse(f_f[:,0], r_f[:,0])
                    y_rmse= util.calculate_mse(f_f[:,1], r_f[:,1])
                    z_rmse= util.calculate_mse(f_f[:,2], r_f[:,2])
                    lose= cri(f_f[:,], r_f[:,])
                    fx_rmse += x_rmse
                    fy_rmse += y_rmse
                    fz_rmse += z_rmse
                    los += lose
                  #  logger.info(f"x_rmse is {x_rmse}")
                  #  logger.info(f"y_rmse is {y_rmse}")
                 #   logger.info(f"z_rmse is {z_rmse}")
                 #   logger.info(f"lose is {lose}")

                fx_rmse = fx_rmse / idx
                fy_rmse = fy_rmse / idx
                fz_rmse = fz_rmse / idx
                los = los / idx

               # time_vla=time.time()-time_vla
               # print("validation")
               # print(time_vla)
               # print("============")

                # log
                logger.info('# Validation # FX_RMSE: {:.4e}'.format(fx_rmse))
                logger.info('# Validation # FY_RMSE: {:.4e}'.format(fy_rmse))
                logger.info('# Validation # FZ_RMSE: {:.4e}'.format(fz_rmse))
                logger.info('# Validation # los: {:.4e}'.format(los))

                logger_val = logging.getLogger('val')  # validation logger
             #   logger_val.info('<epoch:{:3d}, iter:{:8,d}> fx_rmse: {:.4e};fy_rmse: {:.4e};fz_rmse: {:.4e}'.format(
              #      epoch, current_step,fx_rmse,fy_rmse, fz_rmse))
                logger_val.info('<epoch:{:3d}, iter:{:8,d}> fx_rmse: {:.4e};fy_rmse: {:.4e};fz_rmse: {:.4e};los: {:.4e}'.format(
                    epoch, current_step,fx_rmse,fy_rmse, fz_rmse, los))
                # tensorboard logger
                if opt['use_tb_logger'] and 'debug' not in opt['name']:
                    tb_logger.add_scalar('fx_rmse', fx_rmse, current_step)
                    tb_logger.add_scalar('fy_rmse', fy_rmse, current_step)
                    tb_logger.add_scalar('fz_rmse', fz_rmse, current_step)
                    tb_logger.add_scalar('los', los, current_step)



            # save models and training states
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(current_step)
                model.save_training_state(epoch, current_step)

    logger.info('Saving the final model.')
    model.save('latest')
    logger.info('End of training.')


if __name__ == '__main__':
    main()
