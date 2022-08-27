import math
import os
import pathlib
import torch
import torch.nn as nn
from tqdm import tqdm
import random
import Utils.functions as functions
from Models import models_ as models
from Pretrained import vgg
from Utils.dataset import JitDataset
from Utils import PrepareData
import torch.nn.functional as F
import functools
from Utils import losses
from Utils.utils import show_image

def init_models(opt, Height, epochs_list):
    # generator initialization:

    if Height == 0 and opt.use_struct:
        netG = models.PEStructGenerator(opt).to(opt.device)
        netD = models.StructDiscriminator(opt.shapes[0]).to(opt.device)
        epochs_list.append(opt.struct_epochs)
    elif Height in opt.semantic_stages and opt.use_semantic:
        netG = models.SemanticGenerator(opt).to(opt.device)
        netD = models.SemanticDiscriminator(opt).to(opt.device)
        epochs_list.append(opt.semantic_epochs)
    elif Height in opt.dip_stages and opt.use_dip:
        netG = models.DipGenerator(64, 3, [1,1,1], stages = 2, downmode = 'conv', upmode = 'conv', noud = False).to(opt.device)
        netD = models.DipDiscriminator(stage = 1, dim = 64).to(opt.device)
        epochs_list.append(opt.dip_epochs)
    else:
        netG = models.TextureGenerator(opt).to(opt.device)
        netD = models.TextureDiscriminator(opt).to(opt.device)
        epochs_list.append(opt.dip_epochs)
    print(sum([para.numel() for para in netG.parameters()]))
    print(sum([para.numel() for para in netD.parameters()]))
    return netG, netD


class ExSinGAN():
    def __init__(self, opt):
        self.opt = opt

        training_image_name = os.path.basename(opt.input_name).split(".")[0]
        dir2save = 'Result/{}/'.format(training_image_name)
        dir2save += "{}".format(opt.model_name)
        self.opt.dir2save = dir2save
        image_path = os.path.join(opt.input_dir, opt.input_name)
        self.OriginalImage = functions.read_image(image_path ) # 1x3xHxW
        self.real = functions.adjust_scales2image(self.OriginalImage, opt).to(opt.device)  # real: Tensor 1x3x256xW
        self.reals = functions.create_reals_pyramid(self.real, opt)
        self.fixed_probe_noise = torch.randn(14, 1, self.reals[0].shape[-2], self.reals[0].shape[-1]).to(opt.device)
        self.opt.reals = self.reals
        self.PyramidLayers = len(self.reals)
        if not os.path.exists(self.opt.dir2save):
            os.makedirs(self.opt.dir2save)
        functions.save_image(self.OriginalImage, self.opt.dir2save + '/origin.png')
        ###################################
        #####  init PyramidGenerator ######
        ###################################
        self.Gs, self.Ds, self.FixedNoise, self.NoiseAmp, self.start_epoch, self.epochs_list = [], [], [], [], [], []
        self.init_GD()
        self.construct_PyramidGenerator()
        self.resize = functools.partial(F.interpolate, mode = 'bicubic', align_corners = True)

    def init_GD(self):
        for Height in range(self.PyramidLayers):
            self.opt.nfc = min(self.opt.nfc_init * pow(2, math.floor(Height / 4)), 128)
            self.opt.min_nfc = min(self.opt.min_nfc_init * pow(2, math.floor(Height / 4)), 128)
            G, D = init_models(self.opt, Height, self.epochs_list)
            self.Gs.append(G)
            self.Ds.append(D)

    def construct_PyramidGenerator(self):
        self.PyramidGenerator = models.PyramidGenerator(self.Gs, self.reals, self.NoiseAmp, self.opt)

    def load_pretrained(self, model_name, load_D = True):

        path1 = os.path.join(*(self.opt.dir2save.split('/')[:-1]))
        if model_name is not None:
            path2 = model_name
        else:
            path2 = self.opt.dir2save.split('/')[-1]
        self.opt.dir2save = os.path.join(path1, path2)

        for Height in range(self.PyramidLayers):
            if os.path.exists(os.path.join(self.opt.dir2save, str(Height), 'netG.pth')):
                self.Gs[Height].load_state_dict(
                    torch.load(os.path.join(self.opt.dir2save, str(Height), 'netG.pth')))

                D_pth = torch.load(os.path.join(self.opt.dir2save, str(Height), 'netD.pth'))
                if load_D:
                    self.Ds[Height].load_state_dict(D_pth['netD']
                        )
                self.start_epoch.append(D_pth['epoch']+1)
            else:
                self.start_epoch.append(0)

        for Height in range(self.PyramidLayers):
            if self.start_epoch[Height] < self.epochs_list[Height]:
                self.ConstructedLayers = Height
                print(f'Training will start from height {Height}.')
                break
        else:
            self.ConstructedLayers = Height + 1
            print(f'Training completely.')

        if self.ConstructedLayers > 0:
            self.NoiseAmp += torch.load(os.path.join(self.opt.dir2save, 'NoiseAmp.pth'))[:self.ConstructedLayers]
            self.FixedNoise = torch.load(os.path.join(self.opt.dir2save, 'FixedNoise.pth'))

    def continue_train(self, model_name = None):
        self.load_pretrained(model_name)
        if self.ConstructedLayers < self.PyramidLayers:
            self.train()

    def train(self):

        opt = self.opt
        if not os.path.exists(opt.dir2save):
            os.makedirs(opt.dir2save)
        reals = self.reals
        PyramidHeight = self.PyramidLayers

        ############################
        # define FixedNoise for training on reconstruction
        ###########################
        if len(self.FixedNoise) == 0:
            for Height in range(0, PyramidHeight):
                if Height == 0:
                    z_opt = torch.randn(1, 1, reals[Height].shape[2], reals[Height].shape[3]).to(opt.device)
                else:
                    z_opt = torch.zeros_like(reals[Height], device = opt.device)
                self.FixedNoise.append(z_opt.detach())
            torch.save(self.FixedNoise, os.path.join(opt.dir2save, 'FixedNoise.pth'))

        if opt.use_semantic:
            self.vgg = vgg.VGG().to(opt.device)
            self.vgg.load_state_dict(torch.load('Pretrained/' + 'vgg_conv.pth'))

        ###########################
        # START
        ###########################
        for Height in range(self.ConstructedLayers, self.PyramidLayers):
            opt.SubDir2Save = os.path.join(opt.dir2save, str(Height))
            if not os.path.exists(opt.SubDir2Save):
                os.makedirs(opt.SubDir2Save)
            functions.save_image(reals[Height], '{}/real_scale.png'.format(opt.SubDir2Save))

            # self.writer = SummaryWriter(log_dir=opt.SubDir2Save)
            self.CurrentHeight = Height
            if isinstance(self.Gs[Height], models.PEStructGenerator):
                self.train_struct()
            elif isinstance(self.Gs[Height], models.SemanticGenerator):
                self.train_semantic()
            elif isinstance(self.Gs[Height], models.DipGenerator):
                self.train_dip()
            elif isinstance(self.Gs[Height], models.TextureGenerator):
                self.train_texture()
        torch.save(self.Gs, f'{opt.dir2save}/Gs.pth')
        torch.save(self.opt, 'opt.pth')
        print('training completely')

    def train_struct(self):
        Height = self.CurrentHeight
        netG = self.Gs[0]
        netD = self.Ds[0]

        dis_loss = losses.loss_wgan_dis
        gen_loss = losses.loss_wgan_gen

        start_epoch = self.start_epoch[Height]
        if start_epoch == self.epochs_list[Height]:
            return
        opt = self.opt
        real = self.reals[Height]
        D_iters = self.opt.D_iters
        G_iters = self.opt.G_iters
        BATCHSIZE = self.opt.struct_batch_size
        optimizerG = torch.optim.Adam(netG.parameters(), lr = opt.lr_g_struct, betas = (self.opt.beta1, 0.9))
        optimizerD = torch.optim.Adam(netD.parameters(), lr = opt.lr_d_struct, betas = (self.opt.beta1, 0.9))

        SubDir2Save = pathlib.Path(opt.dir2save) / str(Height)
        SubDir2Save.mkdir(parents = False, exist_ok = True)

        alpha = opt.alpha_struct
        if Height == 0:
            self.NoiseAmp.append(1)

        if self.FixedNoise[0] is None:
            self.FixedNoise[0] = torch.randn([1, 1, self.reals[0].shape[2], self.reals[0].shape[3]]).to(opt.device)
            torch.save(self.FixedNoise, os.path.join(opt.dir2save, 'FixedNoise.pth'))

        DT = PrepareData.DataTrans()

        Loader = JitDataset(self.opt.JitDataRoot, image_size = self.reals[0].shape[-2:])
        DataNum = Loader.__len__()
        DataSet = torch.utils.data.DataLoader(Loader, batch_size = DataNum, shuffle = True)
        DataAll = next(iter(DataSet)).to(opt.device)
        DataAll = torch.cat([DataAll, torch.flip(DataAll, dims = [-1])], dim = 0)
        augment_num = int(opt.mix_ratio / (1 - opt.mix_ratio) * DataNum)
        for i in range(augment_num):
            DataAll = torch.cat(
                [DataAll, self.resize(DT.gen(self.real), size = self.reals[0].shape[-2:]).to(self.opt.device)], dim = 0)

        DataNum = DataAll.shape[0]
        DataAll = DataAll[torch.randint(0, DataNum, (DataNum,), dtype = torch.int64)]
        # internal Transform

        EPOCHS = tqdm(range(start_epoch, opt.struct_epochs))

        for epoch in EPOCHS:
            ############################
            # (1) Update D network
            ###########################
            netD.train()
            netG.train()
            gradient_penalty = torch.tensor([0.0]).to(opt.device)
            rec_loss = torch.Tensor([0.0]).to(opt.device)
            diversity_loss = torch.Tensor([0.0]).to(opt.device)
            for i in range(D_iters):
                real_data_v = DataAll[torch.randint(0, DataNum, (BATCHSIZE,), dtype = torch.int64)]
                optimizerD.zero_grad()
                # optimizerDPatch.zero_grad()
                real_data_v = real_data_v.to(opt.device)
                output_real = netD(real_data_v)
                noise = torch.randn(BATCHSIZE, 1, *real_data_v.shape[-2:]).to(opt.device)
                with torch.no_grad():  # totally freeze netG
                    fake = netG(noise)
                output_fake = netD(fake)
                errD_real, errD_fake =  dis_loss(output_fake, output_real)
                gradient_penalty = functions.calc_gradient_penalty_fc(netD, real_data_v, fake)
                D_cost = errD_fake + errD_real + gradient_penalty
                D_cost.backward()
                optimizerD.step()
                # optimizerDPatch.step()

            ############################
            # (2) Update G network
            ###########################
            for i in range(G_iters):
                optimizerG.zero_grad()

                noise = torch.randn(BATCHSIZE, 1, *real_data_v.shape[-2:]).to(opt.device)
                fake = netG(noise)
                output  = netD(fake)
                errG = gen_loss(output)
                diversity_loss = -F.l1_loss(output[0], output[1])
                reconstruction = netG(self.FixedNoise[0].to(opt.device))
                rec_loss = alpha * F.mse_loss(reconstruction, real)
                (errG + rec_loss + diversity_loss).backward()
                optimizerG.step()

            EPOCHS.set_description('stage [{}/{}]:'.format(Height, opt.pyramid_height))
            EPOCHS.set_postfix(Gloss = errG.item(), Dfakeloss = errD_fake.item(), Drealloss = errD_real.item(),
                               gradient_penalty = gradient_penalty.item(), rec_loss = rec_loss.item(), div_loss = diversity_loss.item())
            if (epoch + 1) % 100 == 0 or epoch == 0:
                netG.eval()
                netD.eval()
                functions.save_image(fake.detach(), '{}/fake_sample_{}.png'.format(opt.SubDir2Save, epoch + 1))
                functions.save_image(reconstruction.detach(),
                                     '{}/reconstruction_{}.png'.format(opt.SubDir2Save, epoch + 1))
                torch.save(netG.state_dict(), '%s/netG.pth' % opt.SubDir2Save)
                torch.save({'netD': netD.state_dict(), 'epoch': epoch}, '%s/netD.pth' % opt.SubDir2Save)
                torch.save(self.NoiseAmp, '%s/NoiseAmp.pth' % opt.dir2save)
                self.see_this_height(filename = opt.model_name, height = Height, epoch = epoch)

    def train_semantic(self):
        opt = self.opt
        Height = self.CurrentHeight
        start_epoch = self.start_epoch[Height]
        if start_epoch == self.epochs_list[Height]:
            return
        netG, netD = self.Gs[Height], self.Ds[Height]
        real = self.reals[Height].to(self.opt.device)
        alpha = opt.alpha_semantic
        dis_loss = losses.loss_wgan_dis
        gen_loss = losses.loss_wgan_gen
        perceptual_loss = vgg.perceptual_loss(['r54'], opt.device)

        PrevReconstruction = self.PyramidGenerator(EndHeight = Height - 1, FixedNoise = self.FixedNoise).detach()
        ############################
        # calculate noise_amp
        ###########################
        if Height == 0:
            self.NoiseAmp.append(1)
        else:
            reconstruction = functions.upsampling(PrevReconstruction, real.shape[2], real.shape[3])
            criterion = nn.MSELoss()
            rec_loss = criterion(reconstruction, real)
            RMSE = torch.sqrt(rec_loss).detach()
            _noise_amp = opt.noise_amp_init * RMSE
            self.NoiseAmp.append(_noise_amp)

        optimizerD = torch.optim.Adam(netD.parameters(), lr = opt.lr_d_semantic, betas = (opt.beta1, 0.999))
        optimizerG = torch.optim.Adam(netG.parameters(), lr = opt.lr_g_semantic, betas = (opt.beta1, 0.999))
        schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer = optimizerD, milestones = [1600],
                                                          gamma = opt.gamma)
        schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer = optimizerG, milestones = [1600],
                                                          gamma = opt.gamma)

        real_ = real

        EPOCHs = tqdm(range(start_epoch, opt.semantic_epochs))
        for epoch in EPOCHs:
            netG.train()
            netD.train()
            ############################
            # (1) Update D network: maximize D(x) + D(G(z))
            ###########################
            gradient_penalty = torch.tensor([0.0]).to(opt.device)

            for j in range(2):
                # train with real
                netD.zero_grad()
                output_real = netD(real_)
                with torch.no_grad():
                    PrevImage = self.PyramidGenerator(EndHeight = Height - 1)
                fake = self.PyramidGenerator(StartHeight = Height, EndHeight = Height, PrevImage = PrevImage)
                output_fake = netD(fake.detach())
                errD_real, errD_fake = dis_loss(output_fake, output_real)
                gradient_penalty = functions.calc_gradient_penalty_conv(netD, real, fake)
                errD_total = errD_real + errD_fake + gradient_penalty
                errD_total.backward()
                optimizerD.step()

            ############################
            # (2) Update G network: maximize D(G(z))
            ###########################

            for _ in range(2):
                netG.zero_grad()
                fake = self.PyramidGenerator(StartHeight = Height, EndHeight = Height, PrevImage = PrevImage)
                output = netD(fake)
                errG = gen_loss(output)
                if alpha != 0:
                    loss = nn.MSELoss()

                    reconstruction = self.PyramidGenerator(StartHeight = Height, EndHeight = Height,
                                                       FixedNoise = self.FixedNoise, PrevImage = PrevReconstruction)
                    rec_loss = alpha * loss(reconstruction, real)
                    # else:
                    #     reconstruction = self.PyramidGenerator(StartHeight = Height, EndHeight = Height,
                    #                                            FixedNoise = self.FixedNoise,
                    #                                            PrevImage = torch.flip(PrevReconstruction, dims = [-1]))
                    #     rec_loss = alpha * loss(torch.flip(reconstruction, dims = [-1]), real)

                else:
                    reconstruction = self.PyramidGenerator(StartHeight = Height, EndHeight = Height,
                                                           FixedNoise = self.FixedNoise, PrevImage = PrevReconstruction)
                    rec_loss = torch.Tensor([0]).to(opt.device)

                p_loss = opt.p_weight * perceptual_loss(fake, real) # perceptual_loss(fake, torch.flip(real, dims=[-1])


                errG_total = errG + rec_loss + p_loss
                errG_total.backward()
                optimizerG.step()

            EPOCHs.set_description('stage [{}/{}]:'.format(Height, opt.pyramid_height))
            EPOCHs.set_postfix(Recloss = rec_loss.item(), Perceploss = p_loss.item(), errG = errG.item(),
                              gradient_penalty = gradient_penalty.item(), errD_real = errD_real.item(), errD_fake = errD_fake.item())

            schedulerD.step()
            schedulerG.step()

            if (epoch + 1) % 100 == 0 or epoch == 0:
                netG.eval()
                netD.eval()
                functions.save_image(reconstruction.detach(),
                                     '{}/reconstruction_{}.png'.format(opt.SubDir2Save, epoch + 1))
                torch.save(netG.state_dict(), '%s/netG.pth' % opt.SubDir2Save)
                torch.save({'netD': netD.state_dict(), 'epoch': epoch}, '%s/netD.pth' % opt.SubDir2Save)
                torch.save(self.NoiseAmp, '%s/NoiseAmp.pth' % opt.dir2save)
                self.see_this_height(filename = opt.model_name, height = Height, epoch = epoch)


    def train_dip(self):


        opt = self.opt
        Height = self.CurrentHeight
        start_epoch = self.start_epoch[Height]
        if start_epoch == self.epochs_list[Height]:
            return
        from Utils import losses
        gen_loss = losses.loss_dcgan_gen
        dis_loss = losses.loss_dcgan_dis

        netG, netD = self.Gs[Height], self.Ds[Height]
        real = self.reals[Height].to(self.opt.device)
        alpha = opt.alpha_dip

        PrevReconstruction = self.PyramidGenerator(EndHeight = Height - 1, FixedNoise = self.FixedNoise).detach()

        ############################
        # calculate noise_amp
        ###########################
        if Height == 0:
            self.NoiseAmp.append(1)
        else:
            reconstruction = functions.upsampling(PrevReconstruction, real.shape[2], real.shape[3])
            criterion = nn.MSELoss()
            rec_loss = criterion(reconstruction, real)
            RMSE = torch.sqrt(rec_loss).detach()
            _noise_amp = opt.noise_amp_init * RMSE
            self.NoiseAmp.append(_noise_amp)
        ############################
        # Set optimizer
        ###########################
        optimizerD = torch.optim.Adam(netD.parameters(), lr = 1e-4, betas = (opt.beta1, 0.999))
        optimizerG = torch.optim.Adam(netG.parameters(), lr = 1e-4, betas = (opt.beta1, 0.999))

        rec_epoch = opt.rec_epoch
        real_ = real
        EPOCHs = tqdm(range(start_epoch, opt.dip_epochs))
        gradient_penalty = torch.Tensor([0]).to(opt.device)
        errG = errD_real = errD_fake = torch.Tensor([0.0]).to(opt.device)

        flag = True
        for epoch in EPOCHs:
            netD.train()
            netG.train()
            if epoch > rec_epoch and flag is True:
                optimizerG = torch.optim.Adam(netG.parameters(), lr = 1e-4)
                flag = False
            ############################
            # (1) Update D network: maximize D(x) + D(G(z))
            ###########################
            for j in range(1):
                if epoch > rec_epoch:
                    # train with real
                    optimizerD.zero_grad()
                    output_real = netD(real_.to(opt.device))
                    with torch.no_grad():
                        PrevImage= self.PyramidGenerator(EndHeight = Height - 1)
                        fake = self.PyramidGenerator(StartHeight = Height, EndHeight = Height, PrevImage = PrevImage)
                    output_fake = netD(fake.detach())
                    errD_real, errD_fake = dis_loss(output_fake, output_real)
                    # gradient_penalty = functions.calc_gradient_penalty_conv(netD, real, fake)
                    errD_total = errD_real + errD_fake + gradient_penalty
                    errD_total.backward()
                    optimizerD.step()
            for j in range(1):
                optimizerG.zero_grad()
                PrevReconstruction_noise = torch.clamp(PrevReconstruction + 0.1* torch.randn_like(PrevReconstruction), min = -1, max = 1)
                rec = netG(PrevReconstruction_noise)
                rec_loss = 200 * nn.functional.mse_loss(rec, real)
                if epoch > rec_epoch:
                    with torch.no_grad():
                        PrevImage= self.PyramidGenerator(EndHeight = Height - 1)
                    fake = self.PyramidGenerator(StartHeight = Height, EndHeight = Height, PrevImage = PrevImage)
                    output_fake = netD(fake)
                    errG = gen_loss(output_fake) / 2
                (errG + rec_loss).backward()
                optimizerG.step()

            EPOCHs.set_description('stage [{}/{}]:'.format(Height, opt.pyramid_height))
            EPOCHs.set_postfix(Recloss = rec_loss.item(), Gloss = errG.item(), Dfakeloss = errD_fake.item(),
                              Drealloss = errD_real.item(), gradient_penalty = gradient_penalty.item())




            if (epoch + 1) % 100 == 0 or epoch == 0:
                if epoch > rec_epoch:
                    functions.save_image(fake.detach(), '{}/fake_sample_{}.png'.format(opt.SubDir2Save, epoch + 1))
                functions.save_image(rec.detach(),
                                     '{}/reconstruction_{}.png'.format(opt.SubDir2Save, epoch + 1))
                netG.eval()
                netD.eval()

                torch.save(netG.state_dict(), '%s/netG.pth' % opt.SubDir2Save)
                torch.save({'netD': netD.state_dict(), 'epoch': epoch}, '%s/netD.pth' % opt.SubDir2Save)
                torch.save(self.NoiseAmp, '%s/NoiseAmp.pth' % opt.dir2save)
                self.see_this_height(filename = opt.model_name, height = Height, epoch = epoch, see_pm = False)

    def train_texture(self):
        opt = self.opt
        Height = self.CurrentHeight
        start_epoch = self.start_epoch[Height]
        if start_epoch == self.epochs_list[Height]:
            return
        netG, netD = self.Gs[Height], self.Ds[Height]
        real = self.reals[Height].to(self.opt.device)
        alpha = opt.alpha_texture

        PrevReconstruction = self.PyramidGenerator(EndHeight = Height - 1, FixedNoise = self.FixedNoise).detach()

        ############################
        # calculate noise_amp
        ###########################
        if Height == 0:
            self.NoiseAmp.append(1)
        else:
            reconstruction = functions.upsampling(PrevReconstruction, real.shape[2], real.shape[3])
            criterion = nn.MSELoss()
            rec_loss = criterion(reconstruction, real)
            RMSE = torch.sqrt(rec_loss).detach()
            _noise_amp = opt.noise_amp_init * RMSE
            self.NoiseAmp.append(_noise_amp)
        ############################
        # Set optimizer
        ###########################
        optimizerD = torch.optim.Adam(netD.parameters(), lr = opt.lr_d_texture, betas = (opt.beta1, 0.999))
        optimizerG = torch.optim.Adam(netG.parameters(), lr = opt.lr_g_texture, betas = (opt.beta1, 0.999))
        schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer = optimizerD, milestones = [1600],
                                                          gamma = opt.gamma)
        schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer = optimizerG, milestones = [1600],
                                                          gamma = opt.gamma)

        real_ = real
        _iter = tqdm(range(start_epoch, opt.texture_epochs))
        for epoch in _iter:
            ############################
            # (1) Update D network: maximize D(x) + D(G(z))
            ###########################
            for j in range(opt.Dsteps):
                # train with real
                netD.zero_grad()
                output = netD(real_.to(opt.device))
                errD_real = -output.mean()

                # train with fake
                with torch.no_grad():
                    PrevImage = self.PyramidGenerator(EndHeight = Height - 1)
                fake = self.PyramidGenerator(StartHeight = Height, EndHeight = Height, PrevImage = PrevImage)
                output = netD(fake.detach())
                errD_fake = output.mean()
                gradient_penalty = functions.calc_gradient_penalty_conv(netD, real, fake)
                errD_total = errD_real + errD_fake + gradient_penalty
                errD_total.backward()
                optimizerD.step()

            ############################
            # (2) Update G network: maximize D(G(z))
            ###########################

            for _ in range(opt.Gsteps):
                fake = self.PyramidGenerator(StartHeight = Height, EndHeight = Height, PrevImage = PrevImage)
                output = netD(fake)
                errG = -output.mean()

                if alpha != 0:
                    loss = nn.MSELoss()
                    reconstruction = self.PyramidGenerator(StartHeight = Height, EndHeight = Height,
                                                           FixedNoise = self.FixedNoise, PrevImage = PrevReconstruction)
                    rec_loss = alpha * loss(reconstruction, real)
                else:
                    reconstruction = self.PyramidGenerator(StartHeight = Height, EndHeight = Height,
                                                           FixedNoise = self.FixedNoise, PrevImage = PrevReconstruction)
                    rec_loss = torch.Tensor([0]).to(opt.device)

                netG.zero_grad()
                errG_total = errG + rec_loss
                errG_total.backward()
                optimizerG.step()

            _iter.set_description('stage [{}/{}]:'.format(Height, opt.pyramid_height))
            _iter.set_postfix(Recloss = rec_loss.item(), Gloss = errG.item(), Dfakeloss = errD_fake.item(),
                              DrealScore = -errD_real.item(), gradient_penalty = gradient_penalty.item())

            schedulerD.step()
            schedulerG.step()

            if (epoch+1) % 500 == 0 or epoch == 0:
                functions.save_image(fake.detach(), '{}/fake_sample_{}.png'.format(opt.SubDir2Save, epoch + 1))
                functions.save_image(reconstruction.detach(),
                                     '{}/reconstruction_{}.png'.format(opt.SubDir2Save, epoch+ 1))
                netG.eval()
                netD.eval()

                torch.save(netG.state_dict(), '%s/netG.pth' % opt.SubDir2Save)
                torch.save({'netD': netD.state_dict(), 'epoch': epoch}, '%s/netD.pth' % opt.SubDir2Save)
                torch.save(self.NoiseAmp, '%s/NoiseAmp.pth' % opt.dir2save)
                self.see_this_height(filename = opt.model_name, height = Height, epoch = epoch)

    def random_generate(self, filename = None):
        opt = self.opt
        if filename is None:
            filename = self.opt.timestamp
        for height in range(0, 1):
            SubSaveDir = os.path.join(opt.dir2save, 'Random')
            if not os.path.exists(SubSaveDir):
                os.makedirs(SubSaveDir)
            for i in range(self.opt.sample_num):
                if height == 0:
                    Img = self.PyramidGenerator(StartHeight = height)
                else:
                    Img = self.PyramidGenerator(StartHeight = height, PrevImage = self.reals[height - 1])
                functions.save_image(Img, os.path.join(SubSaveDir, f'{i}.jpg'))
        print('ramdom sample has completed')

    def see_this_height(self, filename = None, height = 0, epoch = 0, suffix = 'sample', see_pm = False):
        opt = self.opt
        if filename is None:
            filename = self.opt.timestamp

        SubSaveDir = opt.SubDir2Save
        if not os.path.exists(SubSaveDir):
            os.makedirs(SubSaveDir)

        for i in range(1):
            Img = self.PyramidGenerator(EndHeight = height - 1, start_noise = self.fixed_probe_noise)
            functions.save_image(Img, os.path.join(SubSaveDir, f'{suffix}{i}_lr.png'),
                                 nrow = self.fixed_probe_noise.shape[0] // 2)
            Img = self.PyramidGenerator(EndHeight = height, start_noise = self.fixed_probe_noise)
            functions.save_image(Img, os.path.join(SubSaveDir, f'{suffix}{i}_epoch{epoch}.png'), nrow = self.fixed_probe_noise.shape[0] // 2)
            if see_pm is True and height == opt.pyramid_height:


                functions.save_image(self.Gs[-1](Img), os.path.join(SubSaveDir, f'{suffix}{i}_epoch{epoch}_wopm.png'),
                                     nrow = self.fixed_probe_noise.shape[0] // 2)

                Img,_ = self.PyramidGenerator.ttsr(Img, self.reals[height - 1], self.reals[height - 1],
                          fold_params = self.PyramidGenerator.fold_params,
                          divisor = self.PyramidGenerator.divisor, n = 3, lv = 1, skip = 4, return_img = True)
                functions.save_image(Img, os.path.join(SubSaveDir, f'{suffix}{i}_lr_pm.png'),
                                     nrow = self.fixed_probe_noise.shape[0] // 2)




