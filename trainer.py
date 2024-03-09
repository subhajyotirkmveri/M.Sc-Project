from collections import defaultdict
import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm, trange
import torchvision
import utils
from torch.utils.tensorboard import SummaryWriter

class perceptionLoss():
    def __init__(self, args):
        vgg = torchvision.models.vgg19(pretrained=True)
        vgg.eval()
        self.features = vgg.features.to(args.device)
        self.feature_layers = ['4', '9', '18', '27', '36']
        self.mse_loss = nn.MSELoss()
"""__init__(self, args): This is the constructor method of the perceptionLoss class. It initializes the perception loss object. Inside the constructor:

It loads the VGG-19 model pretrained on ImageNet using torchvision.models.vgg19(pretrained=True).
Sets the model to evaluation mode using vgg.eval().
Stores the features of the VGG model in the self.features attribute.
Specifies the layers of interest from which features will be extracted using self.feature_layers.
Initializes the Mean Squared Error (MSE) loss function using nn.MSELoss() and stores it in self.mse_loss."""

    def getfeatures(self, x):
        feature_list = []
        for name, module in self.features._modules.items():
            x = module(x)
            if name in self.feature_layers:
                feature_list.append(x)
        return feature_list
"""Initialization: It initializes an empty list feature_list to store the extracted features.

Feature Extraction Loop: It iterates over the modules of the VGG-19 model stored in self.features. Each module represents a layer of the VGG-19 network.

Forward Pass: For each module, it applies the module to the input tensor x, thereby passing the input through that layer and obtaining the output feature map.

Filtering by Layer: It checks if the name of the current module is present in self.feature_layers. If it is, it means that the current layer's feature map needs to be extracted based on the specified layers.

Storing Features: If the name of the module is in self.feature_layers, it appends the output feature map x to the feature_list.

Returning Features: Finally, it returns the list of extracted feature maps feature_list. Each element of this list corresponds to the feature map extracted from a layer specified in self.feature_layers"""

    def calculatePerceptionLoss(self, video_pd, video_gt):
        features_pd = self.getfeatures(video_pd.view(video_pd.size(0)*video_pd.size(1), video_pd.size(2), video_pd.size(3), video_pd.size(4)))
        features_gt = self.getfeatures(video_gt.view(video_gt.size(0)*video_gt.size(1), video_gt.size(2), video_gt.size(3), video_gt.size(4)))
        
        with torch.no_grad():
            features_gt = [x.detach() for x in features_gt]
        
        perceptual_loss = sum([self.mse_loss(features_pd[i], features_gt[i]) for i in range(len(features_gt))])
        return perceptual_loss
"""Extract Features:
It extracts features from the predicted video (video_pd) and the ground truth video (video_gt) using the getfeatures method. The videos are reshaped into a format suitable for processing by a neural network.

Detach Ground Truth Features:
The features extracted from the ground truth video are detached from the computation graph using torch.no_grad() and list comprehension. This ensures that gradients are not computed for these features during backpropagation.

Compute Loss:
It calculates the mean squared error (MSE) loss between corresponding pairs of features from the predicted and ground truth videos.
The MSE loss is computed for each pair of features using self.mse_loss.
The individual losses are summed up to get the overall perception loss."""
class tfaceTrainer:
    def __init__(self, 
                args, 
                generator,
                disc_frame,
                disc_pair,
                disc_emo,
                disc_video,
                train_loader,
                val_loader):
        
        self.args = args
        
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.generator = generator

        self.disc_frame = disc_frame
        self.disc_pair = disc_pair
        self.disc_emo = disc_emo
        self.disc_video = disc_video

        # self.plotter = utils.VisdomLinePlotter(env=args.env_name)
        self.plotter = SummaryWriter(args.out_path)

        self.l1_loss = torch.nn.L1Loss()
        self.emo_loss = nn.CrossEntropyLoss()

        self.emo_loss_disc = nn.CrossEntropyLoss()
        self.loss_dict = defaultdict(list)

        self.global_step = 0

        self.perception_loss = perceptionLoss(args)
"""Initializes the trainer object with various components required for training.
Parameters:
args: The configuration arguments for the training process.
generator: The TFace generator model.
disc_frame, disc_pair, disc_emo, disc_video: The discriminator models for different components of the TFace model.
train_loader, val_loader: Data loaders for the training and validation datasets.
plotter: The object for plotting and visualization during training.
l1_loss: The L1 loss function used for certain loss calculations.
emo_loss: The cross-entropy loss function used for emotion prediction.
emo_loss_disc: The cross-entropy loss function used for emotion discrimination.
loss_dict: A defaultdict to store the losses during training.
global_step: A variable to keep track of the current step in training.
perception_loss: An instance of the perceptionLoss class, used for computing the perception loss during training."""    
    def freezeNet(self, network):
        for p in network.parameters():
            p.requires_grad = False
    
    def unfreezeNet(self, network):
        for p in network.parameters():
            p.requires_grad = True

"""freezeNet(self, network):

Freezes the parameters of the given network by setting requires_grad attribute to False for all parameters.
unfreezeNet(self, network):

Unfreezes the parameters of the given network by setting requires_grad attribute to True for all parameters."""

    def schdulerStep(self):
        self.generator.module.scheduler.step()
        if self.args.disc_pair:
            self.disc_pair.module.scheduler.step()
        if self.args.disc_frame:
            self.disc_frame.module.scheduler.step()
        if self.args.disc_video:
            self.disc_video.module.scheduler.step()
        if self.args.disc_emo:
            self.disc_emo.module.scheduler.step()

    def displayLRs(self):
        lr_list = [self.generator.module.opt.param_groups]
        if self.args.disc_pair:
            lr_list.append(self.disc_pair.module.opt.param_groups)
        if self.args.disc_frame:
            lr_list.append(self.disc_frame.module.opt.param_groups)
        if self.args.disc_video:
            lr_list.append(self.disc_video.module.opt.param_groups)
        if self.args.disc_emo:
            lr_list.append(self.disc_emo.module.opt.param_groups)
        
        cnt = 0
        for lr in lr_list:
            for param_group in lr:
                print('LR {}: {}'.format(cnt, param_group['lr']))
                cnt+=1


"""schdulerStep(self):

This method steps the learning rate scheduler for each network if it is defined. It adjusts the learning rate according to the predefined schedule during training.
displayLRs(self):

This method displays the learning rates (LRs) of each optimizer associated with the networks being trained. It retrieves the LR from each parameter group within the optimizer and prints it out, along with an index for identification."""

    def saveNetworks(self, fold):
        torch.save(self.generator.state_dict(), os.path.join(self.args.out_path, fold, 'generator.pt'))
        if self.args.disc_pair:
            torch.save(self.disc_pair.state_dict(), os.path.join(self.args.out_path, fold, 'disc_pair.pt'))
        if self.args.disc_frame:
            torch.save(self.disc_frame.state_dict(), os.path.join(self.args.out_path, fold, 'disc_frame.pt'))
        if self.args.disc_video:
            torch.save(self.disc_video.state_dict(), os.path.join(self.args.out_path, fold, 'disc_video.pt'))
        if self.args.disc_emo:
            torch.save(self.disc_emo.state_dict(), os.path.join(self.args.out_path, fold, 'disc_emo.pt'))
        print('Networks has been saved to {}'.format(fold))


"""saveNetworks(self, fold):

This method saves the state dictionaries of the generator and discriminator networks to disk. It takes a fold argument representing the directory to save the networks.
It saves each network if it is defined in the configuration (args)."""

    def calcGANLoss(self, logit, label):
        if label == 'real':
            return -logit.mean()
        if label == 'fake':
            return logit.mean()

"""calcGANLoss(self, logit, label):

This method calculates the loss for a GAN (Generative Adversarial Network) based on the logits and the label ('real' or 'fake').
If the label is 'real', it returns the negative mean of the logits.
If the label is 'fake', it returns the mean of the logits."""


    def logLosses(self, t):
        desc_str=''
        for key in sorted(self.loss_dict.keys()):
            desc_str += key + ': %.5f' % (np.nanmean(self.loss_dict[key])) + ', '
        t.set_description(desc_str)

"""logLosses(self, t):

This method logs the losses accumulated during training. It formats the loss values as a string and sets it as the description of a tqdm progress bar (t).
The formatted string includes the name of each loss and its average value."""


    def plotLosses(self, var_name, xlabel, ylabel, legend, title, rem=0):
        if self.global_step%self.args.plot_interval == rem:
            for key in legend:
                try:
                    self.plotter.add_scalar("Loss/train", self.loss_dict[key][-1], self.global_step)
                except:
                    continue
 
 """plotLosses(self, var_name, xlabel, ylabel, legend, title, rem=0):

This method plots the losses during training using a tensorboard SummaryWriter (plotter).
It adds the current loss value to the tensorboard under the tag "Loss/train" along with the global step.
It plots the losses only if the current global step is divisible by the plot interval (args.plot_interval), with the remainder equal to rem."""       
        # Visdom Plotter
        # if self.global_step%self.args.plot_interval == rem:
        #     x = []
        #     y = []
        #     for key in legend:
        #         y.append(np.nanmean(self.loss_dict[key][-5:]))
        #         x.append(self.global_step)
        #     self.plotter.plot(var_name, xlabel, ylabel, legend, title, x, y)

    def convertVid(self, V):
        return (0.5 + (V/2.0))


"""def convertVid(self, V)::

This line defines a method named convertVid that takes a tensor V as input.
return (0.5 + (V/2.0)):

This line returns a tensor obtained by normalizing the input tensor V to the range [0, 1]. It adds 0.5 to each element of V and then divides by 2."""
    def logValImages(self, epoch):
        speech_v, video_v, att_v, emotion_v = [d.float().to(self.args.device) for d in next(iter(self.val_loader))]
        self.generator.eval()
        pd_video_v, z_spch_v, emo_label_v = self.generator(video_v[:, np.random.randint(video_v.shape[1], size=1)[0], ...], speech_v, emotion_v)

        pd_video_v = pd_video_v[:, :, :, :, :]
        video_v_p = video_v[:, :, :, :, :]

        pd_video_v = pd_video_v.view(pd_video_v.size(0) * pd_video_v.size(1), pd_video_v.size(2), pd_video_v.size(3), pd_video_v.size(4))
        video_v_p = video_v_p.view(video_v_p.size(0) * video_v_p.size(1), video_v_p.size(2), video_v_p.size(3), video_v_p.size(4))

        grid = torchvision.utils.make_grid(self.convertVid( torch.cat((pd_video_v[:, :, :, :], video_v_p[:, :, :, :]), 0) ))
        self.plotter.add_image("Predicted and GT Video Frames", grid, self.global_step)
        
        # Visdom Plotter
        # self.plotter.viz.images( self.convertVid( torch.cat((pd_video_v[:, :, :, :], video_v_p[:, :, :, :]), 0) ), 
        #                         opts=dict(jpgquality=70, store_history=False, caption='e'+str(epoch)+"_check_"+str(self.global_step),title='e'+str(epoch)+"_check_"+str(self.global_step)),
        #                         env=self.args.env_name,
        #                         win='samples',
        #                         nrow=self.args.num_frames,
        #                         )



"""def logValImages(self, epoch)::

This line defines a method named logValImages that logs validation images during training. It takes an epoch argument indicating the current epoch.
speech_v, video_v, att_v, emotion_v = [d.float().to(self.args.device) for d in next(iter(self.val_loader))]:

This line loads a batch of validation data from the validation loader (val_loader) and converts each element of the batch to a float tensor before moving it to the specified device (self.args.device).
self.generator.eval():

This line sets the generator network to evaluation mode, which disables dropout and batch normalization layers.
pd_video_v, z_spch_v, emo_label_v = self.generator(video_v[:, np.random.randint(video_v.shape[1], size=1)[0], ...], speech_v, emotion_v):

This line generates predicted video frames (pd_video_v), speech embeddings (z_spch_v), and emotion labels (emo_label_v) using the generator network. It takes a random frame from the input video (video_v), along with speech and emotion data as inputs to the generator.
pd_video_v = pd_video_v[:, :, :, :, :]:

This line ensures that the predicted video frames (pd_video_v) have a consistent shape for further processing.
video_v_p = video_v[:, :, :, :, :]:

This line ensures that the original video frames (video_v_p) have a consistent shape for further processing.
pd_video_v = pd_video_v.view(pd_video_v.size(0) * pd_video_v.size(1), pd_video_v.size(2), pd_video_v.size(3), pd_video_v.size(4)):

This line reshapes the predicted video frames tensor (pd_video_v) to have a flattened batch dimension for visualization.
video_v_p = video_v_p.view(video_v_p.size(0) * video_v_p.size(1), video_v_p.size(2), video_v_p.size(3), video_v_p.size(4)):

This line reshapes the original video frames tensor (video_v_p) to have a flattened batch dimension for visualization.
grid = torchvision.utils.make_grid(self.convertVid( torch.cat((pd_video_v[:, :, :, :], video_v_p[:, :, :, :]), 0) )):

This line concatenates the predicted video frames and the original video frames along the batch dimension, converts the resulting tensor to the [0, 1] range using the convertVid method, and creates a grid of images using make_grid function from torchvision.utils.
self.plotter.add_image("Predicted and GT Video Frames", grid, self.global_step):

This line adds the generated grid of images (predicted and ground truth video frames) to the tensorboard SummaryWriter (plotter) under the tag "Predicted and GT Video Frames" at the current global step."""


    def step_disc_frame(self, data):
        self.disc_frame.train()
        speech, video_gt, mrm, emotion, image_c, video_pd = data
        self.disc_frame.module.opt.zero_grad()

"""self.disc_frame.train():

Sets the discriminator network (disc_frame) to training mode.
speech, video_gt, mrm, emotion, image_c, video_pd = data:

Unpacks the elements of the input data, which presumably contains speech data (speech), ground truth video frames (video_gt), motion representation models (mrm), emotion labels (emotion), conditioned image data (image_c), and predicted video frames (video_pd).
self.disc_frame.module.opt.zero_grad():

Zeroes out the gradients of the discriminator's optimizer to prepare for the backward pass."""
        logit_fake = self.disc_frame(image_c, video_pd)
        logit_real = self.disc_frame(image_c, video_gt)

        loss_fake = self.calcGANLoss(logit_fake, 'fake')
        loss_real = self.calcGANLoss(logit_real, 'real')

        self.loss_dict['loss_df_fake'].append(loss_fake.item())
        self.loss_dict['loss_df_real'].append(loss_real.item())

        gp, grad_norm = self.disc_frame.module.compute_grad_penalty(video_gt, video_pd, image_c)
        
        self.loss_dict['df_gp'].append(gp.item())
        self.loss_dict['df_gnorm'].append(grad_norm.item())
"""
logit_fake = self.disc_frame(image_c, video_pd) and logit_real = self.disc_frame(image_c, video_gt):

Passes both the predicted and ground truth video frames along with conditioned images through the discriminator network to obtain the logits for fake and real videos.
loss_fake = self.calcGANLoss(logit_fake, 'fake') and loss_real = self.calcGANLoss(logit_real, 'real'):

Calculates the GAN loss for the fake and real video logits using the calcGANLoss method.
gp, grad_norm = self.disc_frame.module.compute_grad_penalty(video_gt, video_pd, image_c):

Computes the gradient penalty (gp) and gradient norm (grad_norm) for the discriminator using real and fake video frames and conditioned images."""
        loss = loss_fake + loss_real + self.args.disc_frame_gp*gp

        wdistance = -(loss_fake + loss_real).item()

        self.loss_dict['df_wdistance'].append(wdistance)

        loss.backward()
        self.disc_frame.module.opt.step()

        self.plotLosses('Disc Frame Losses', 'iterations', 'loss', ['loss_df_fake', 'loss_df_real'], 'Disc Frame Losses', rem=1)
        self.plotLosses('frame_wdistance', 'iterations', 'loss', ['df_wdistance'], 'wdistance', rem=1)
        self.plotLosses('frame_gp', 'iterations', 'loss', ['df_gp', 'df_gnorm'], 'gp', rem=1)


"""loss = loss_fake + loss_real + self.args.disc_frame_gp*gp:

Calculates the total loss for the discriminator, which includes the GAN losses for fake and real videos and the gradient penalty term.
wdistance = -(loss_fake + loss_real).item():

Calculates the Wasserstein distance as the negative sum of the GAN losses for fake and real videos.
loss.backward():

Backpropagates the total loss through the discriminator network.
self.disc_frame.module.opt.step():

Updates the discriminator's parameters using the optimizer.
self.plotLosses(...):

Logs and plots the discriminator's losses, Wasserstein distance, and gradient penalty during training for visualization and monitoring purposes."""

    def step_disc_emo(self, data):
        self.disc_emo.train()
        speech, video_gt, mrm, emotion, image_c, video_pd = data
        self.disc_emo.module.opt.zero_grad()

        class_fake = self.disc_emo(image_c, video_pd)
        class_real = self.disc_emo(image_c, video_gt)
"""self.disc_emo.train():

Sets the emotion discriminator (disc_emo) to training mode.
speech, video_gt, mrm, emotion, image_c, video_pd = data:

Unpacks the elements of the input data, which presumably contains speech data (speech), ground truth video frames (video_gt), mouth region mask (mrm), emotion labels (emotion), conditioned image data (image_c), and predicted video frames (video_pd).
self.disc_emo.module.opt.zero_grad():

Zeroes out the gradients of the emotion discriminator's optimizer to prepare for the backward pass.
class_fake = self.disc_emo(image_c, video_pd) and class_real = self.disc_emo(image_c, video_gt):

Passes both the predicted and ground truth video frames along with conditioned images through the emotion discriminator network to obtain the predicted emotion classes."""
        loss_fake_c = self.emo_loss_disc(class_fake, (6*torch.ones_like(torch.argmax(emotion, dim=1))).long().to(self.args.device))
        loss_real_c = self.emo_loss_disc(class_real, torch.argmax(emotion, dim=1))

        self.loss_dict['loss_fake_c'].append(loss_fake_c.item())
        self.loss_dict['loss_real_c'].append(loss_real_c.item())

        loss = 0.5*(loss_fake_c + loss_real_c) 

        loss.backward()
        self.disc_emo.module.opt.step()

        self.plotLosses('Disc Emotion', 'iterations', 'loss', ['loss_fake_c', 'loss_real_c'], 'disc_emo', rem=1)
        
"""loss_fake_c = self.emo_loss_disc(class_fake, (6*torch.ones_like(torch.argmax(emotion, dim=1))).long().to(self.args.device)) and loss_real_c = self.emo_loss_disc(class_real, torch.argmax(emotion, dim=1)):

Calculates the emotion classification loss for the fake and real video frames using the predicted and ground truth emotion labels. The fake loss uses a fixed label corresponding to 'neutral' emotion.
loss = 0.5*(loss_fake_c + loss_real_c):

Calculates the total emotion discriminator loss as the average of the fake and real losses.
loss.backward():

Backpropagates the total loss through the emotion discriminator network.
self.disc_emo.module.opt.step():

Updates the emotion discriminator's parameters using the optimizer.
self.plotLosses('Disc Emotion', 'iterations', 'loss', ['loss_fake_c', 'loss_real_c'], 'disc_emo', rem=1):

Logs and plots the emotion discriminator's losses during training for visualization and monitoring purposes."""

    def step_disc_emo_recog(self, data):
        self.disc_emo.train()
        speech, video_gt, mrm, emotion, image_c = data
        self.disc_emo.module.opt.zero_grad()
 
        class_real = self.disc_emo(image_c, video_gt)

        loss = self.emo_loss_disc(class_real, torch.argmax(emotion, dim=1))
"""self.disc_emo.train():

Sets the emotion discriminator (disc_emo) to training mode.
speech, video_gt, mrm, emotion, image_c = data:

Unpacks the elements of the input data, which presumably contains speech data (speech), ground truth video frames (video_gt), motion representation models (mrm), emotion labels (emotion), and conditioned image data (image_c).
self.disc_emo.module.opt.zero_grad():

Clears the gradients of the emotion discriminator's parameters to prepare for the backward pass.
class_real = self.disc_emo(image_c, video_gt):

Passes the conditioned image and ground truth video frames through the emotion discriminator network to obtain the predicted emotion classes.
loss = self.emo_loss_disc(class_real, torch.argmax(emotion, dim=1)):

Calculates the emotion classification loss using the predicted emotion classes and the ground truth emotion labels."""
        self.loss_dict['loss_classifier'].append(loss.item())

        loss.backward()
        self.disc_emo.module.opt.step()

        self.plotLosses('Disc Emo Losses', 'iterations', 'loss', ['loss_classifier'], 'Disc Emo Losses')
"""self.loss_dict['loss_classifier'].append(loss.item()):

Appends the computed loss to the loss_classifier key in the loss dictionary for logging and visualization.
loss.backward():

Backpropagates the loss through the emotion discriminator network.
self.disc_emo.module.opt.step():

Updates the emotion discriminator's parameters based on the computed gradients.
self.plotLosses('Disc Emo Losses', 'iterations', 'loss', ['loss_classifier'], 'Disc Emo Losses'):

Logs and plots the emotion discriminator's losses during training, specifically the classification loss, for monitoring and analysis."""

    def step_generator(self, data):
        if self.args.disc_frame:
            self.disc_frame.eval()
            self.freezeNet(self.disc_frame)
        if self.args.disc_emo:
            # self.disc_emo.eval()
            self.freezeNet(self.disc_emo)

        self.generator.train()
        speech, video_gt, mrm, emotion, image_c = data
        self.generator.module.opt.zero_grad()

        video_pd, z_spch, emo_label = self.generator(image_c, speech, emotion)
        
        if self.args.disc_frame:
            df = self.disc_frame.forward(image_c, video_pd)
            loss_df = self.calcGANLoss(df, 'real')
        if self.args.disc_emo:
            de_c = self.disc_emo.forward(image_c, video_pd)
            loss_de_c = self.emo_loss(de_c, torch.argmax(emotion, dim=1))
            self.loss_dict['loss_de_c'].append(loss_de_c.item())

        perception_loss = self.perception_loss.calculatePerceptionLoss(video_pd, video_gt)

        recon_loss = 100*self.l1_loss(video_pd*mrm, video_gt*mrm)

        emo_loss = self.emo_loss(emo_label, torch.argmax(emotion, dim=1))

        self.loss_dict['loss_rec'].append(recon_loss.item())
        self.loss_dict['loss_emo'].append(emo_loss.item())
        self.loss_dict['perception_loss'].append(perception_loss.item())        

        loss = 0.001*emo_loss + recon_loss + perception_loss
        if self.args.disc_frame:
            loss += self.args.disc_frame * loss_df
        if self.args.disc_emo:
            loss_demo = self.args.disc_emo * loss_de_c
            self.loss_dict['loss_demo'].append(loss_demo.item())
            loss += loss_demo

        self.loss_dict['loss_gen'].append(loss.item())

        loss.backward()
        self.generator.module.opt.step()

        if self.args.disc_frame:
            self.unfreezeNet(self.disc_frame)
        if self.args.disc_emo:
            self.unfreezeNet(self.disc_emo)
            self.plotLosses('Gen Emotion', 'iterations', 'loss', ['loss_de_c'], 'gen_emo')
        self.plotLosses('Gen Losses', 'iterations', 'loss', ['loss_rec', 'loss_gen', 'perception_loss', 'loss_demo'], 'Gen Losses')
        

    def train(self):
        for epoch in tqdm(range(self.args.num_epochs)):
            diterator = iter(self.train_loader)
            # with trange(1) as t:
            with trange(len(self.train_loader)) as t:     
                for i in t:               
                    speech, video, mrm, emotion = [d.float().to(self.args.device) for d in next(diterator)]

                    mrm = mrm.unsqueeze(2)
                    mrm = mrm + 0.01

                    rnd_idx = 0
                    # rnd_idx = np.random.randint(video.shape[1], size=1)[0] # Using first frame of the sequence provides better results, using random images might be more robust
                    image_c = video[:, rnd_idx, :, :, :]

                    data = [speech, video, mrm, emotion, image_c]
                    
                    if self.global_step%2 == 0:
                        self.step_generator(data)
                    elif self.global_step%2 == 1:
                        with torch.no_grad():
                            if self.args.disc_pair or self.args.disc_frame or self.args.disc_video or self.args.disc_emo:
                                video_pd, _, _ = self.generator(image_c, speech, emotion)
                                video_pd = video_pd.detach()
                                data = [speech, video, mrm, emotion, image_c, video_pd]

                        if self.args.disc_frame:
                            self.step_disc_frame(data)

                        if self.args.disc_emo:
                            self.step_disc_emo(data)
                            
                    if self.global_step % 50 == 0:
                        self.logValImages(epoch)
                        self.saveNetworks('inter')

                    self.global_step += 1
            
            self.schdulerStep()
            self.displayLRs()

            self.saveNetworks('')


    def pre_train(self):
        for epoch in tqdm(range(self.args.num_epochs)):
            diterator = iter(self.train_loader)
            with trange(len(self.train_loader)) as t:     
                for i in t:               
                    speech, video, mrm, emotion = [d.float().to(self.args.device) for d in next(diterator)]
                    mrm = mrm.unsqueeze(2)
                    mrm = mrm + 0.01

                    rnd_idx = 0
                    image_c = video[:, rnd_idx, :, :, :]

                    data = [speech, video, mrm, emotion, image_c]
                    
                    self.step_disc_emo_recog(data)

                    self.logLosses(t)

                    if self.global_step % 500 == 0:
                        self.saveNetworks('inter')

                    self.global_step += 1
            
            self.schdulerStep()
            self.displayLRs()

            self.saveNetworks('')
            
