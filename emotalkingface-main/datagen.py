import os
import numpy as np
import h5py  # Imports the h5py library, which is a Python interface to the HDF5 binary data format.
import random as rn

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
import librosa  #Imports the librosa library, which is used for audio processing and analysis.
import cv2
import albumentations as A #Imports the albumentations library and aliases it as A, which is a popular library for image augmentation.
from torchvision import transforms
from scipy.spatial import procrustes
from scipy import signal
import scipy.ndimage.filters as fi #Imports the filters module from the scipy.ndimage library and aliases it as fi, which provides various image filtering functions

def gkern2(means, nsig=9):  #Defines a function gkern2 that takes two parameters: means (a list or array containing two values) and nsig (the standard deviation of the 
    """Returns a 2D Gaussian kernel array."""#Gaussian filter, default is 9). The function returns a 2D Gaussian kernel array.
    
    inp = np.zeros((128, 128)) #Creates a 2D NumPy array filled with zeros, with dimensions 128x128, and assigns it to the variable inp.
    
    if int(means[1]) > 127 or int(means[0]) > 127:
        inp[92, 92] = 1
    else:
        inp[int(means[1]), int(means[0])] = 1
    return fi.gaussian_filter(inp, nsig)

emotion_dict = {'ANG':0, 'DIS':1, 'FEA':2, 'HAP':3, 'NEU':4, 'SAD':5}
intensity_dict = {'XX':0, 'LO':1, 'MD':2, 'HI':3}

class DatasetContainer():
    def __init__(self, args, val=False): # args, which is a set of arguments, and val, a boolean parameter with a default value of False 
        self.args = args
        self.filelist = []  # Initializes an empty list called filelist to store information about files in the dataset.

        if not val:        #If val is False (default), it uses self.args.in_path, likely the directory containing the training data
            path = self.args.in_path  # If val is True, it uses self.args.val_path, likely the directory containing the validation data.
        else:
            path = self.args.val_path

        for root, dirnames, filenames in os.walk(path):
            for filename in filenames:
                if os.path.splitext(filename)[1] == '.hdf5':
                    labels = os.path.splitext(filename)[0].split('_')
                    emotion = emotion_dict[labels[2]]
                    
                    emotion_intensity = intensity_dict[labels[3]]
                    if val:
                        if emotion_intensity != 3:
                            continue
                    
                    self.filelist.append((root, filename, emotion, emotion_intensity))

        self.filelist = np.array(self.filelist)
        print('Num files: ', len(self.filelist))

    def getDset(self):
        return FaceDset(self.filelist, self.args)  #Returns a FaceDset object initialized with the filelist and arguments. The FaceDset object likely represents a 
                                                      #dataset class used for facial recognition tasks.
class FaceDset(Dataset): #Defines a class named FaceDset that inherits from torch.utils.data.Dataset.

    def __init__(self, filelist, args):# Defines the initialization method of the FaceDset class. It takes two arguments: filelist, which contains information about 
        self.filelist = filelist        #files in the dataset, and args, a set of arguments.
        self.args = args
        
        self.transform = transforms.Compose([transforms.ToTensor()]) # Initializes a transformation pipeline using torchvision.transforms.Compose that converts input 
                                                                      #images to tensors.
        target = {}
        for i in range(1, self.args.num_frames):  # Iterates over the range of numbers from 1 to the number of frames specified in args.
            target['image' + str(i)] = 'image'  #Creates keys in the target dictionary for each frame, mapping them to the string 'image'

        self.augments = A.Compose([
                        A.RandomBrightnessContrast(p=0.2),    
                        A.RandomGamma(p=0.2),    
                        A.CLAHE(p=0.2),
                        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=50, val_shift_limit=50, p=0.2),  
                        A.ChannelShuffle(p=0.2), 
                        A.RGBShift(p=0.2),
                        A.RandomBrightness(p=0.2),
                        A.RandomContrast(p=0.2),
                        # A.HorizontalFlip(p=0.5),            # Initializes a separate augmentation pipeline specifically for channel-wise augmentations, including 
                        A.GaussNoise(var_limit=(10.0, 50.0), p=0.25)  #adding Gaussian noise. The p parameter specifies the probability of applying the augmentations.
                    ], additional_targets=target, p=0.8)

        self.c_augments = A.Compose([A.GaussNoise(p=1),
            ], p=0.5)

        self.normTransform = A.Compose([ 
                        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), always_apply=True)
                    ], additional_targets=target, p=1)
        
    def __len__(self):
        return len(self.filelist)

    # def normFrame(self, frame):
    #     normTransform = self.normTransform(image=frame)
    #     frame = self.transform(normTransform['image'])
    #     return frame

    def normFrame(self, frame):
        normTransform = self.normTransform(image=frame)
        frame = normTransform['image']
        frame = np.moveaxis(frame, 2, 0)
        return torch.from_numpy(frame)

"""This line moves the axis of the normalized frame using np.moveaxis. The original frame format is assumed to be in the channels-last format (height, width, channels), but PyTorch expects the channels-first format (channels, height, width). So, this line rearranges the axes to match PyTorch's format."""

    def augmentVideo(self, video):
        args = {}
        args['image'] = video[0, :, :, :] #The first frame of the video (index 0) is extracted and assigned to the key 'image' in the args dictionary.
        for i in range(1, self.args.num_frames):
            args['image' + str(i)] = video[i, :, :, :] # For each frame index i, the corresponding frame is extracted from the video and assigned to a key 'image{i}' 
        result = self.augments(**args)                  #in the args dictionary.
        video[0, :, :, :] = result['image']
        for i in range(1, self.args.num_frames):
            video[i, :, :, :] = result['image' + str(i)]
        return video
"""result = self.augments(**args): The data augmentation transformation defined in the augments attribute is applied to the video frames using the arguments stored in the args dictionary. The **args syntax unpacks the dictionary and passes its key-value pairs as keyword arguments to the augments function.

video[0, :, :, :] = result['image']: After augmentation, the transformed first frame is retrieved from the result dictionary and assigned back to the first frame of the original video.

for i in range(1, self.args.num_frames):: Another loop iterates over the remaining frames of the video.

video[i, :, :, :] = result['image' + str(i)]: For each frame index i, the corresponding transformed frame is retrieved from the result dictionary and assigned back to the original video."""

    def __getitem__(self, idx):

        filename = self.filelist[idx]
        emotion = int(filename[2])
        emotion = to_categorical(emotion, num_classes=6)
        emotion_intensity = int(filename[3]) # We don't use this info
            
        filename = filename[:2]
       
        dset = h5py.File(os.path.join(*filename), 'r')
        try:
            idx = np.random.randint(dset['video'].shape[0]-self.args.num_frames, size=1)[0]

"""Randomly selects a start index within the video sequence to avoid reaching the end of the video. If there's an exception (likely due to the dataset being smaller than the expected shape), it recursively calls itself with a random index."""
        except:
            return self.__getitem__(np.random.randint(len(self.filelist)-1, size=1)[0])
     
        video = dset['video'][idx:idx+self.args.num_frames, :, :, :]  #Extracts the video frames and landmark coordinates corresponding to the selected index from the 
        lmarks = dset['lmarks'][idx:idx+self.args.num_frames, 48:, :] # the dataset
    
        lmarks = np.mean(lmarks, axis=1)  #Averages the landmark data across the second dimension (likely averaging multiple landmark points for each frame).
        video = self.augmentVideo(video) #Performs data augmentation (e.g., random cropping, flipping) on the video frames using the augmentVideo function (explained 
                                         #earlier).
        att_list = []  #Creates an empty list to store attention masks.
        video_normed = []
        for i in range(video.shape[0]):
            video_normed.append(self.normFrame(video[i, :, :, :]))
            att = gkern2(lmarks[i, :]) # Likely calculates an attention mask (att) based on the averaged landmarks for the current frame (lmarks[i, :]) using a 
                                       #function named gkern2
            att = att / np.max(att)
            att_list.append(att)

"""att_list = []: Initializes an empty list to store the attention maps generated for each frame in the video.
video_normed = []: Initializes an empty list to store the normalized video frames.
for i in range(video.shape[0]):: Iterates over each frame in the video.
video_normed.append(self.normFrame(video[i, :, :, :])): Calls the normFrame method to normalize the current video frame and appends it to the video_normed list.
att = gkern2(lmarks[i, :]): Generates a 2D Gaussian kernel as an attention map based on the landmark coordinates of the current frame.
att = att / np.max(att): Normalizes the attention map to ensure its values lie within the range [0, 1].
att_list.append(att): Appends the normalized attention map to the att_list list for later use."""

        video_normed = torch.stack(video_normed, 0) 
#video_normed = torch.stack(video_normed, 0): Converts the list of normalized video frames into a PyTorch tensor and stacks them 
                                         #along a new dimension (dimension 0), creating a batch of normalized video frames.
        att_list = np.array(att_list) 

 
        speech = dset['speech'][:]
        speech = speech/np.max(np.abs(speech))
        
        speech = speech[ int(idx*self.args.increment): int((idx+self.args.num_frames)*self.args.increment)]
#Extracts the speech data corresponding to the current video frame sequence based on the index and frame increment.
        speech = np.reshape(speech, (1, speech.shape[0])) 
#Reshapes the speech data into a 2D array with one row and the number of columns equal to the length of the speech sequence.        
        if speech.shape[1] != self.args.increment*self.args.num_frames:
            return self.__getitem__(np.random.randint(len(self.filelist)-1, size=1)[0])
 #Checks if the length of the speech sequence matches the expected length based on the number of frames and increment. If not, it recursively calls the _getitem_ #method to fetch data from another random index.            

        return speech, video_normed, att_list, emotion
#Returns the speech data, normalized video frames, attention maps, and emotion label for the current sample.

def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)
    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    # Example
    ```python
    # Consider an array of 5 labels out of a set of 3 classes {0, 1, 2}:
    > labels
    array([0, 2, 1, 2, 0])
    # `to_categorical` converts this into a matrix with as many
    # columns as there are classes. The number of rows
    # stays the same.
    > to_categorical(labels)
    array([[ 1.,  0.,  0.],
           [ 0.,  0.,  1.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.],
           [ 1.,  0.,  0.]], dtype=float32)
    ```
    """

    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

"""y = np.array(y, dtype='int'): Converts the input y (class vector) into a NumPy array with integer data type.
input_shape = y.shape: Retrieves the shape of the input array y.
if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:: Checks if the input shape is valid for conversion.
input_shape = tuple(input_shape[:-1]): Updates the input shape if necessary.
y = y.ravel(): Flattens the input array y.
if not num_classes:: Checks if the number of classes is provided as an argument. If not, it determines the number of classes based on the maximum value in y.
n = y.shape[0]: Retrieves the number of elements in the flattened array y.
categorical = np.zeros((n, num_classes), dtype=dtype): Initializes an empty binary matrix with dimensions (n, num_classes).
categorical[np.arange(n), y] = 1: Sets the appropriate elements in the binary matrix to 1 based on the class labels in y.
output_shape = input_shape + (num_classes,): Calculates the shape of the output matrix.
categorical = np.reshape(categorical, output_shape): Reshapes the binary matrix to the desired output shape.
return categorical: Returns the binary matrix representation of the input class vector."""

