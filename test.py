import os
import sys
import logging
import time
import argparse
import numpy as np
from collections import OrderedDict
import torch.nn as nn
import options.options as option
import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model
import torch.nn as nn
# options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
util.mkdirs((path for key, path in opt['path'].items() if not key == 'pretrain_model_G'))
opt = option.dict_to_nonedict(opt)

util.setup_logger(None, opt['path']['log'], 'test.log', level=logging.INFO, screen=True)
logger = logging.getLogger('base')
logger.info(option.dict2str(opt))
# Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

# Create model
model = create_model(opt)

for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt['name']
    logger.info('\nTesting [{:s}]...'.format(test_set_name))
    test_start_time = time.time()
  #  dataset_dir = os.path.join(opt['path']['results_root'], test_set_name)
 #   util.mkdir(dataset_dir)

    test_results = OrderedDict()
    test_results['fz_rmse'] = []
    test_results['fy_rmse'] = []
    test_results['fx_rmse'] = []
    test_results['pinjia'] = []
    test_results['loss'] = []

  #  test_results['ssim_y'] = []

    for data in test_loader:
     #   need_HR = False if test_loader.dataset.opt['dataroot_HR'] is None else True

        model.feed_data(data)
        img_path = data['img_path'][0]
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        model.test()  # test
        force = model.get_current_force()
        logger.info(f"fake is {force['f_f']}")
        logger.info(f"real is {force['r_f']}")
        cri=nn.MSELoss()
        loss=cri(force['f_f'],force['r_f'])
        test_results['loss'].append(loss)
        logger.info(f"loss is {loss}")

        fx_rmse=util.calculate_mse(force['f_f'][:,0], force['r_f'][:,0])
        test_results['fx_rmse'].append(fx_rmse)
        fy_rmse=util.calculate_mse(force['f_f'][:,1], force['r_f'][:,1])
        test_results['fy_rmse'].append(fy_rmse)
        fz_rmse=util.calculate_mse(force['f_f'][:,2], force['r_f'][:,2])
        test_results['fz_rmse'].append(fz_rmse)
        pinjia=(fx_rmse/force['r_f'][:,0]+fy_rmse/force['r_f'][:,1]+fz_rmse/force['r_f'][:,2])/3


        logger.info(' --- fx_rmse: {:.6f}; fy_rmse: {:.6f};fz_rmse: {:.6f}.'.format(fx_rmse, fy_rmse, fz_rmse))

    ave_loss= sum(test_results['loss']) / len(test_results['loss'])
    ave_fxrm= sum(test_results['fx_rmse']) / len(test_results['fx_rmse'])
    ave_fyrm= sum(test_results['fy_rmse']) / len(test_results['fy_rmse'])
    ave_fzrm= sum(test_results['fz_rmse']) / len(test_results['fz_rmse'])

    logger.info('----Average  fx_rmse/fy_rmse/fz_rmse/loss results for ----\n\t fx_mse: {:.6f};fy_mse: {:.6f};fz_mse: {:.6f};loss: {:.6f}\n'.format(ave_fxrm,ave_fyrm,ave_fzrm,ave_loss))
                

