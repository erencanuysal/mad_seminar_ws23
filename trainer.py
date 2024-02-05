from models.modules import  ConvNet64,DeConvNet64,GivenAE_encoder,GivenAE_decoder
from models.nae import NAE_L2_OMI
from torch.optim import SGD, Adam
from tqdm import tqdm
from metrics import averageMeter
from torchvision.utils import make_grid, save_image
import torchvision.transforms as transforms
from PIL import Image
import time
import os
import numpy as np
import cv2
import pickle
from utils.mcmc import LangevinSampler
from sklearn.metrics import roc_auc_score
import torch
import matplotlib.pyplot as plt

class NAE_Trainer():
    """feed-forward energy-based model"""

    def __init__(
        self,
        config,
        device,
        train_data_module
    ):
        super().__init__()  
        self.device=device 
        self.config=config
        self.train_data_module=train_data_module
        if config['arch']=='convnet': 
            self.encoder = ConvNet64(
                    in_chan=config['in_chan'],
                    out_chan=config['out_chan'],
                    nh=32,
                    num_groups=None,
                    use_bn=False,
                    out_activation='linear',
                    activation='relu'
                )
            self.decoder =  net = DeConvNet64(
                    in_chan=config['out_chan'], out_chan=config['in_chan'], nh=8,num_groups=None,use_bn=False, out_activation='sigmoid'
                )
        elif config['arch']=='given':
            self.encoder = GivenAE_encoder()
            self.decoder = GivenAE_decoder()
        self.sampler_z=LangevinSampler(n_step= config['sampler']['z_step'],
            stepsize=  config['sampler']['z_stepsize'],
            noise_std= 0.05,
            noise_anneal= None,
            clip_langevin_grad= None,
            buffer_size= 10000,
            replay_ratio= 0.95,
            mh= False,
            bound= 'spherical', 
            initial_dist= 'uniform_sphere')
        self.sampler_x= LangevinSampler(n_step=  config['sampler']['x_step'],
            stepsize=  config['sampler']['x_stepsize'],
            noise_std= 0.05,
            noise_anneal= 1,
            clip_langevin_grad= 0.01,
            mh= False,
            buffer_size= 0,
            bound= [0, 1])
        self.model =  NAE_L2_OMI(self.encoder, self.decoder, self.sampler_z, self.sampler_x, gamma= 1, l2_norm_reg_de= None, l2_norm_reg_en= 0.0001, T= 1.).to('cuda')
    def save_list_to_file(data_list, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(data_list, file)
    def roc_btw_arr(arr1, arr2):
        true_label = np.concatenate([np.ones_like(arr1),
                                    np.zeros_like(arr2)])
        score = np.concatenate([arr1, arr2])
        return roc_auc_score(true_label, score)

    def predict( m, dl, device, flatten=False):
        """run prediction for the whole dataset"""
        l_result = []
        for x in dl:
            with torch.no_grad():
                if flatten:
                    x = x.view(len(x), -1)
                pred = m.predict(x.cuda(device)).detach().cpu()
            l_result.append(pred)
        return torch.cat(l_result)

    def save_model(model, best=False, i_iter=None, i_epoch=None):
        logdir="./logdir/"
        if best:
            pkl_name = "model_best.pkl"
        else:
            if i_iter is not None:
                pkl_name = "model_iter_{}.pkl".format(i_iter)
            else:
                pkl_name = "model_epoch_{}.pkl".format(i_epoch)
        state = {"epoch": i_epoch, "model_state": model.state_dict(), 'iter': i_iter}
        save_path = os.path.join(logdir, pkl_name)
        torch.save(state, save_path)
        print(f'Model saved: {pkl_name}')

    def train(self):
        print_interval= 100
        val_interval= 5
        save_interval= 2000
        save_interval_epoch= 2000
        ae_lr= self.config['ae_lr']
        nae_lr= self.config['nae_lr']
        temperature_lr= 1.0e-3
        print_interval_nae=10
        load_ae=self.config['load_ae']
        best_val_loss = np.inf
        time_meter = averageMeter()
        i = 0
        indist_train_loader = self.train_data_module.train_dataloader()
        #print(indist_train_loader.shape)
        indist_val_loader = self.train_data_module.val_dataloader()
        no_best_model_tolerance = 3
        no_best_model_count = 0

        n_ae_epoch = 0
        n_nae_epoch = 15

        ae_opt = ae_opt = Adam(self.model.parameters(), lr=ae_lr)
        l_params = [{'params': list(self.model.encoder.parameters()) + list(self.model.decoder.parameters())}]
        nae_opt = Adam(l_params, lr=nae_lr)

        #AE run
        if load_ae==True:
                state_dict = torch.load("./logdir/model_epoch_115.pkl")['model_state']
                encoder_state = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder')}
                decoder_state = {k.replace('decoder.', ''): v for k, v in state_dict.items() if k.startswith('decoder')}
                self.model.encoder.load_state_dict(encoder_state)
                self.model.decoder.load_state_dict(decoder_state)
                print('model loaded from')
        train_loss=[]
        val_loss=[]
        for i_epoch in range(n_ae_epoch):
            d_train=0
            for x in indist_train_loader:
                i += 1

                self.model.train()
                x = x.to(self.device)
                #y = y.to(device)

                start_ts = time.time()
                d_train = self.model.train_step_ae(x, ae_opt)
                time_meter.update(time.time() - start_ts)
                train_loss.append(d_train['loss'])
            print("loss: ",d_train['loss'])
            print(i_epoch,". epoch")

            if i % val_interval == 1:
                self.model.eval()
                for val_x in indist_val_loader:
                    val_x = val_x.to(self.device)

                    d_val = self.model.validation_step_ae(val_x, y=None)
                    #print("validation loss:",d_val['loss'])
                    val_loss.append(d_val['loss'])
            if i_epoch % save_interval_epoch == 1:
                self.save_model(self.model, best=False, i_iter=None, i_epoch=i_epoch)
                print(f'Epoch [{i_epoch:d}] model saved {save_interval_epoch}')

            if no_best_model_count > no_best_model_tolerance:
                print('terminating autoencoder training since validation loss does not decrease anymore')
                break

        '''NAE PASS'''
        i = 0
        for i_epoch in tqdm(range(n_nae_epoch)):
            for x in tqdm(indist_train_loader):
                i += 1

                x = x.to(self.device)
                # from pudb import set_trace; set_trace()
                d_result = self.model.train_step(x, nae_opt)
                print("train loss:",d_result['loss'])
                train_loss.append(d_result['loss'])
                #logger.process_iter_train(d_result)
                if i % val_interval==1:
                    self.model.eval()
                    for val_x in indist_val_loader:
                        val_x = val_x.to(self.device)

                        d_val = self.model.validation_step_ae(val_x, y=None)
                        #print("validation loss:",d_val['loss'])
                        val_loss.append(d_val['loss'])
                if i % print_interval_nae == 1:
                    input_img = make_grid(x.detach().cpu(), nrow=10)
                    recon_img = make_grid(self.model.reconstruct(x).detach().cpu(), nrow=10)

                    grid_image_input = transforms.ToPILImage()(input_img).convert("RGB")
                    grid_image_recon = transforms.ToPILImage()(recon_img).convert("RGB")
                    grid_image_input.save('input_image.png')
                    grid_image_recon.save('recon_image.png')
                    #logger.d_train['input_img@'] = input_img
                    #logger.d_train['recon_img@'] = recon_img
                    #logger.summary_train(i)

        #        if i % val_interval == 1:
        #           '''AUC'''
        #           in_pred = predict(model, indist_val_loader, device)
        #           ood1_pred = predict(model, oodval_val_loader, device)
        #           auc_val = roc_btw_arr(ood1_pred, in_pred)
        #           ood2_pred = predict(model, oodtarget_val_loader, device)
        #           auc_target = roc_btw_arr(ood2_pred, in_pred)
        #           d_result = {'nae/auc_val_': auc_val, 'nae/auc_target_': auc_target}
        #           #logger.process_iter_val(d_result)
        #           #print(logger.summary_val(i)['print_str'])
        #           torch.save({'model_state': model.state_dict()}, f'{logdir}/nae_iter_{i}.pkl')
            #torch.save(model.state_dict(), f'./logdir/nae_{i_epoch}.pkl')
        #torch.save(model.state_dict(), f'./logdir/nae.pkl')

        '''AUC'''
        #in_pred = predict(model, indist_val_loader, device)
        #ood1_pred = predict(model, oodval_val_loader, device)
        #auc_val = roc_btw_arr(ood1_pred, in_pred)
        #ood2_pred = predict(model, oodtarget_val_loader, device)
        #auc_target = roc_btw_arr(ood2_pred, in_pred)
        #d_result = {'nae/auc_val': auc_val, 'nae/auc_target': auc_target}
        #print(d_result)

        print("Training finished ")