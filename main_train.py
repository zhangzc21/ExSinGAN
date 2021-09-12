import os
import shutil

import numpy as np

from BigGAN.RunJitter import RunJitter
from Models.ExSinGAN import ExSinGAN
from Models.config import GetReady

if __name__ == '__main__':

    opt = GetReady().opt
    if opt.model_name is None:
        opt.model_name = f'height{opt.pyramid_height}'
        if opt.use_struct:
            opt.model_name += f'struct{opt.struct_channel_dim}bs{opt.struct_batch_size}'
        if opt.use_semantic:
            opt.model_name += '_semantic{}'.format(len(opt.semantic_stages))
        if opt.use_texture:
            opt.model_name += '_texture{}'.format(opt.pyramid_height - len(opt.semantic_stages))

    if opt.use_struct is True:
        opt.JitDataRoot = 'JitData/' + opt.input_name.split('.')[0]
        if not os.path.exists(opt.JitDataRoot):
            os.makedirs(opt.JitDataRoot)
        if len(os.listdir(opt.JitDataRoot)) == 0:
            RunJitter('%s/%s' % (opt.input_dir, opt.input_name), stds = opt.stds, num_sample = opt.jitter_num,
                      cls = opt.cls)

    if opt.ref_struct is not None:
        main_path = os.path.join('TrainedModels', opt.input_name.split('.')[0])
        if os.path.exists(main_path):
            if len(os.listdir(main_path)) > 0:
                G_path = os.path.join(main_path, opt.ref_struct, '0', 'netG.pth')
                D_path = os.path.join(main_path, opt.ref_struct, '0', 'netD.pth')
                if os.path.exists(G_path):
                    if not os.path.exists(os.path.join(main_path, opt.model_name, '0')):
                        os.makedirs(os.path.join(main_path, opt.model_name, '0'))
                    shutil.copy(G_path, os.path.join(main_path, opt.model_name, '0', 'netG.pth'))
                    shutil.copy(D_path, os.path.join(main_path, opt.model_name, '0', 'netD.pth'))
                else:
                    print('The ref doesn\'t exists.')
    config = vars(opt)
    print(config)
    print(opt.input_name)
    model = ExSinGAN(opt)
    np.save(os.path.join(model.opt.dir2save, 'config.npy'), config)

    model.continue_train(model_name = opt.model_name)
    model.random_generate(filename = opt.model_name)
    print("______END______")
