import os
import torch
from Models.ExSinGAN import ExSinGAN
from Utils import functions
import argparse
import numpy as np


class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def random_samples(model_path, num = 50, scale_h = 1, scale_w = 1):
    config_path = os.path.join(model_path, 'config.npy')
    config = np.load(config_path, allow_pickle = True).item()
    opt = Dict(config)
    opt.sample_num = num
    model = ExSinGAN(opt)
    reals = model.reals
    reals_ = [torch.nn.functional.interpolate(real, scale_factor = (scale_h, scale_w)) for real in reals]
    model.reals = reals_
    model.Gs = torch.load(model.opt.dir2save + '/Gs.pth')
    model.NoiseAmp = torch.load(model.opt.dir2save + '/NoiseAmp.pth')
    model.construct_PyramidGenerator()

    StartHeight = 0

    if scale_w == 1.0 and scale_h == 1.0:
        SavePath = os.path.join(opt.out, opt.input_name.split('.')[0], 'Random', opt.model_name, str(StartHeight))
    else:
        SavePath = os.path.join(opt.out, opt.input_name.split('.')[0], 'Random', opt.model_name,
                                f'h{scale_h}w{scale_w}', str(StartHeight))


    if not os.path.exists(SavePath):
        os.makedirs(SavePath)
    import time
    st = time.time()
    for i in range(num):
        Img = model.PyramidGenerator(StartHeight = StartHeight)
        # functions.save_image(Img, os.path.join(SavePath, f'{i}.png'))
    et = time.time()
    print((et-st)/num,'s')



if __name__ == '__main__':

    model = 'TrainedModels/balloons/conv'
    # for h in [0.5, 0.75, 1.25, 1.5]:
    #     scale_h = h
    #     scale_w = 1
    #     parser = argparse.ArgumentParser()
    #     parser.add_argument('--num', type = int, default = 50)
    #     parser.add_argument('--model_path', type = str, default = model)
    #     parser.add_argument('--scale_h', type = float, default = scale_h)
    #     parser.add_argument('--scale_w', type = float, default = scale_w)
    #
    #     Param = parser.parse_args()
    #     random_samples(Param.model_path, Param.num, Param.scale_h, Param.scale_w)

    # for w in [0.5, 0.75, 1.25, 1.5]:
    scale_h = 1
    scale_w = 1
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type = int, default = 100)
    parser.add_argument('--model_path', type = str, default = model)
    parser.add_argument('--scale_h', type = float, default = scale_h)
    parser.add_argument('--scale_w', type = float, default = scale_w)

    Param = parser.parse_args()
    random_samples(Param.model_path, Param.num, Param.scale_h, Param.scale_w)
