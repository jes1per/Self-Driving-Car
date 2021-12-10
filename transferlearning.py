# %%
from skimage.transform import resize
import numpy as np
from itertools import product
from collections import OrderedDict
import os
#from tqdm.notebook import tqdm
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, utils, models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from Run import RunBuilder as RB
from Run import RunManager as RM
from DataLoading import UdacityDataset as UD
from DataLoading import ConsecutiveBatchSampler as CB

from model import transferlearning_regnet_x as TL

# %run Visualization/Visualization.ipynb

torch.set_printoptions(linewidth=120) 
device = torch.device("cuda:0")

def show_sample(sample,pred_angle):
    r""" Helper function for (batch) sample visualization
    Args:
        sample: Dictionary
    """
    image_dims = len(sample['image'].shape)
    assert image_dims <= 5, "Unsupported image shape: {}".format(sample['image'].shape)
    if image_dims == 4:
        error = abs(pred_angle - sample['angle'])
        n = sample['image'].shape[0]
        sample['image'] = torch.Tensor(resize(sample['image'], (n,3,480,640),anti_aliasing=True))
        images = sample['image'].permute(0,2,3,1)
        fig = plt.figure(figsize=(155, 135))
        for i in range(n):
            ax = fig.add_subplot(10,5,i+1)
            ax.imshow(images[i])
            ax.axis('off')
            ax.set_title("t={}".format(sample['timestamp'][i]))
            ax.text(10, 30, sample['frame_id'][i], color='red')
            ax.text(10, 390, "error {:.3}".format(error[i].item()), color='red')
            ax.text(10, 430, "man-angle {:.3}".format(sample['angle'][i].item()), color='red')
            ax.text(10, 470, "pred-angle {:.3}".format(pred_angle[i].item()), color='red')
    else:
        #error = abs(pred_angle - sample['angle'])
        sample['image'] = sample['image'].permute(0,2,1,3,4)
        batch_size,seq_len,channel = sample['image'].shape[0],sample['image'].shape[1],sample['image'].shape[2]
        sample['image'] = torch.Tensor(resize(sample['image'], (batch_size,seq_len,3,480,640),anti_aliasing=True))
        n0 = sample['image'].shape[0]
        n1 = sample['image'].shape[1] if image_dims == 5 else 1
        images_flattened = torch.flatten(sample['image'], end_dim=-4)
        fig, ax = plt.subplots(n0, n1, figsize=(25, 15))
        for i1 in range(n1):
            for i0 in range(n0):
                image = images_flattened[i0 * n1 + i1]
                axis = ax[i0, i1]
                axis.imshow(image.permute(1,2,0))
                axis.axis('off')
                axis.set_title("t={}".format(sample['timestamp'][i0][i1]))
                axis.text(10, 30, sample['frame_id'][i0][i1], color='red')
                #axis.text(10, 390, "error {:.3}".format(error[i0][i1].item()), color='red')
                axis.text(10, 430, "man-angle {:.3}".format(sample['angle'][i0][i1].item()), color='red')
                axis.text(10, 470, "pred-angle {:.3}".format(pred_angle[i0][i1].item()), color='red')
    

def visualize_cnn(cnn, max_row=10, max_col=10, select_channel=None, tb_writer=None):
    assert isinstance(cnn, nn.Conv3d) or isinstance(cnn, nn.Conv2d)
    
    filters = cnn.weight.cpu().detach().numpy() # output_ch x input_ch x D x H x W
    output_ch = np.minimum(cnn.weight.shape[0], max_col)
    input_ch = np.minimum(cnn.weight.shape[1], max_row)
    print(filters.shape)
    
    if isinstance(cnn, nn.Conv3d):
        if select_channel:
            assert isinstance(select_channel, int)
            filters = filters[:, :, select_channel, :, :]
        else:
            filters = np.mean(filters[:, :, :, :, :], axis=2)
    
    plt_idx = 0
    fig = plt.figure(figsize=(output_ch, input_ch))
    plt.xlabel("Output Channel")
    plt.ylabel("Input Channel")
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    frame1.axes.yaxis.set_ticklabels([])
    for o in range(output_ch):
        for i in range(input_ch):
            image = filters[o, i, :, :]
            plt_idx += 1
            ax = fig.add_subplot(input_ch, output_ch, plt_idx)
            ax.imshow(image)
            ax.axis('off')
#     plt.tight_layout()
    plt.show()


class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, register_hooks=True):
        self.model = model
        self.grad = None
        self.conv_output = None
        
        if register_hooks:
            self.register_hooks()
        
    def gradient_hook(self, model, grad_input, grad_output):
        self.grad = grad_output[0].cpu().detach().numpy()
        
    def conv_output_hook(self, model, input, output):
        self.conv_output = output.cpu().detach().numpy()
        
    def register_hooks(self):
        raise NotImplementedError("You should implement this method for your own model!")
        
    def forward(self, x):
        raise NotImplementedError("You should implement this method for your own model!")
        
    def to_image(self, height=None, width=None):
        assert self.grad is not None and self.conv_output is not None, "You should perform both forward pass and backward propagation first!"
        # both grad and conv_output should have the same dimension of: (*, channel, H, W)
        # we produce image(s) of shape: (*, H, W)
        channel_weight = np.mean(self.grad, axis=(-2, -1)) # *, channel
        conv_permuted = np.moveaxis(self.conv_output, [-2, -1], [0, 1]) # H, W, *, channel
        cam_image_permuted = channel_weight * conv_permuted # H, W, *, channel
        cam_image_permuted = np.mean(cam_image_permuted, axis=-1) # H, W, *
        cam_image = np.moveaxis(cam_image_permuted, [0, 1], [-2, -1]) # *, H, W
        
        if height is not None and width is not None:
            image_shape = list(cam_image.shape)
            image_shape[-2] = height
            image_shape[-1] = width
            cam_image = resize(cam_image, image_shape)
        return cam_image
        
    
class CamExtractorTLModel(CamExtractor):
    
    def register_hooks(self):
        self.model.ResNet.layer4.register_forward_hook(self.conv_output_hook)
        self.model.ResNet.layer4.register_backward_hook(self.gradient_hook)

class CamExtractor3DCNN(CamExtractor):
    
    def gradient_hook(self, model, grad_input, grad_output):
        grad = grad_output[0].cpu().detach().numpy()
        self.grad = np.moveaxis(grad, 1, 2) # restore old dimension (batch x seq_len x channel x H x W)
        
    def conv_output_hook(self, model, input, output):
        conv_output = output.cpu().detach().numpy()
        self.conv_output = np.moveaxis(conv_output, 1, 2) # restore old dimension (batch x seq_len x channel x H x W)
    
    def register_hooks(self):
        self.model.Convolution6.register_forward_hook(self.conv_output_hook)
        self.model.Convolution6.register_backward_hook(self.gradient_hook)
    

# %% [markdown]
# # Model Training / Loading

# %%
parameters = OrderedDict(
    learning_rate = [0.001],
    batch_size = [50],
    num_workers = [1],
    #shuffle = [True,False]
)

m = RM.RunManager()
for run in RB.RunBuilder.get_runs(parameters):
    network = TL.TLearning_regnetx()
    network.cuda()
    network.to(device)
    optimizer = optim.Adam(network.parameters(),lr = run.learning_rate,betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001/run.batch_size, amsgrad=False)
# modify the time to visible time format, also use it to check the sequency of pictures is from former to current
    udacity_dataset = UD.UdacityDataset(csv_file='/home/kxk190041/data/self_driving/train/interpolated.csv',
                                     root_dir='/home/kxk190041/data/self_driving/train/',
                                     transform=transforms.Compose([transforms.ToTensor()]),
                                     select_camera='center_camera')
    dataset_size = int(len(udacity_dataset))
    del udacity_dataset
    split_point = int(dataset_size * 0.8)

    training_set = UD.UdacityDataset(csv_file='/home/kxk190041/data/self_driving/train/interpolated.csv',
                                     root_dir='/home/kxk190041/data/self_driving/train/',
                                     transform=transforms.Compose([transforms.ToTensor()]),
                                     select_camera='center_camera',
                                     select_range=(0,split_point))

    validation_set = UD.UdacityDataset(csv_file='/home/kxk190041/data/self_driving/train/interpolated.csv',
                                     root_dir='/home/kxk190041/data/self_driving/train/',
                                     transform=transforms.Compose([transforms.ToTensor()]),
                                     select_camera='center_camera',
                                     select_range=(split_point,dataset_size))
    print("size of training set :{}".format(len(training_set)))
    print("size of validation set :{}".format(len(validation_set)))
    
    training_cbs = CB.ConsecutiveBatchSampler(data_source=training_set, batch_size=run.batch_size, shuffle=True, drop_last=False, seq_len=1)
    training_loader = DataLoader(training_set, sampler=training_cbs, num_workers=run.num_workers, collate_fn=(lambda x: x[0]))

    validation_cbs = CB.ConsecutiveBatchSampler(data_source=validation_set, batch_size=run.batch_size, shuffle=True, drop_last=False, seq_len=1)
    validation_loader = DataLoader(validation_set, sampler=validation_cbs, num_workers=run.num_workers, collate_fn=(lambda x: x[0]))

    training_time = 0

    m.begin_run( run,network,[run.batch_size,3,224,224] )
    for epoch in range(10):
        st_time = time.time()
        m.begin_epoch()
        
        for training_sample in tqdm(training_loader):
            training_sample['image'] = training_sample['image'].squeeze()
            training_sample['image'] = torch.Tensor(resize(training_sample['image'], (run.batch_size,3,224,224),anti_aliasing=True))
            param_values = [v for v in training_sample.values()]
            image,angle = param_values[0],param_values[3]
            image = image.to(device)
            #print(image.dtype)
            #image = image.to(torch.float)
            prediction = network(image)
            prediction = prediction.to(device)
            labels = angle.to(device)
            labels = labels.to(torch.float)
            del param_values, image, angle
            if labels.shape[0]!=prediction.shape[0]:
                prediction = prediction[-labels.shape[0],:]
            training_loss_angle = F.mse_loss(prediction,labels,size_average=None, reduce=None, reduction='mean')
            optimizer.zero_grad()# zero the gradient that are being held in the Grad attribute of the weights
            training_loss_angle.backward() # calculate the gradients
            optimizer.step() # finishing calculation on gradient 
            break
        print("Done")
        training_time += (time.time() - st_time) 
# Calculation on Validation Loss
        with torch.no_grad():    
            for Validation_sample in tqdm(validation_loader):
                Validation_sample['image'] = Validation_sample['image'].squeeze()
                Validation_sample['image'] = torch.Tensor(resize(Validation_sample['image'], (run.batch_size,3,224,224),anti_aliasing=True))

                param_values = [v for v in Validation_sample.values()]
                image,angle = param_values[0],param_values[3]
                image = image.to(device)
                prediction = network(image)
                prediction = prediction.to(device)
                labels = angle.to(device)
                labels = labels.to(torch.float)
                del param_values, image, angle
                if labels.shape[0]!=prediction.shape[0]:
                    prediction = prediction[-labels.shape[0],:]
                validation_loss_angle = F.mse_loss(prediction,labels,size_average=None, reduce=None, reduction='mean')
                m.track_loss(validation_loss_angle, labels.shape[0])
                m.track_num_correct(prediction,labels) 
        m.end_epoch()
        torch.save(network.state_dict(), "/home/kxk190041/Self-Driving-Car/results/Angle_Adam_MSE_Paper_Model-epoch-{}".format(epoch))
    m.end_run()
m.save('result')
print('Training time taken: ', training_time)

# %%
# Load Directly from disk

tl_model = TL.TLearning_EffNetB7().to(device)
tl_model.load_state_dict(torch.load('/home/kxk190041/Self-Driving-Car/results/Angle_Adam_MSE_Paper_Model-epoch-9'))

# %% [markdown]
# # Visualization

# %%
visualize_cnn(tl_model.ResNet.conv1)

# %% [markdown]
# ### GradCAM

# %%
udacity_dataset = UD.UdacityDataset(csv_file='/home/kxk190041/data/self_driving/train/interpolated.csv',
                                 root_dir='/home/kxk190041/data/self_driving/train/',
                                 transform=transforms.Compose([transforms.ToTensor()]),
                                 select_camera='center_camera')


# Load arbitrary data
sample = udacity_dataset[3693]
show_sample(sample)
input_image = sample['image'].reshape(-1, 3, 480, 640).cuda()

cam_extractor_tl = CamExtractorTLModel(tl_model)

# Forward pass
model_output = tl_model(input_image)

# Backward pass
tl_model.zero_grad()
mse_loss = nn.MSELoss()
loss = mse_loss(model_output, sample['angle'].cuda().reshape(1,1))
loss.backward()

cam_image = cam_extractor_tl.to_image(height=480, width=640) # Use this line to extract CAM image from the model!
plt.imshow(cam_image[0, :, :], cmap='jet', alpha=0.5) # this shows CAM as overlay to the original input image


# %%



