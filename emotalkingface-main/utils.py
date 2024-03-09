import matplotlib as mpl   ## Imports the matplotlib library with the alias mpl
mpl.use('Agg')  ## Configures matplotlib to use the 'Agg' backend, which is suitable for non-interactive plotting
import matplotlib.pyplot as plt  ## Imports the pyplot module from matplotlib with the alias plt. This module provides a MATLAB-like interface for creating plots and visualizations.
from mpl_toolkits import mplot3d ##  Imports the mplot3d module from mpl_toolkits, which provides tools for creating 3D plots.
import matplotlib.animation as manimation  ## Imports the animation module from matplotlib, which provides support for creating animations
import matplotlib.lines as mlines  ##  Imports the mlines module from matplotlib, which provides classes for representing and drawing lines.
from matplotlib.lines import Line2D  ##  
from matplotlib import transforms  ## Imports the transforms module from matplotlib, which provides utilities for transforming graphical objects.
import matplotlib.gridspec as gridspec
import argparse, os, fnmatch, shutil
import numpy as np
import cv2  ## Imports the cv2 module, which is the OpenCV library for computer vision and image processing.
import math
import copy
#import librosa  # lrpDisabled as below it is no longer supporting .sound from v0.8
import soundfile as sf # lrpAdd to replace librosa

# import dlib
import subprocess  ##  Imports the subprocess module, which allows you to spawn new processes, connect to their input/output/error pipes, and obtain their return codes.
from tqdm import tqdm  ##  Imports the tqdm module, which provides a fast, extensible progress bar for loops and iterables.
import visdom

font = {'size'   : 18}  ## Defines a dictionary font with a single key 'size' set to 18.
mpl.rc('font', **font)  ## Configures the font settings for matplotlib using the dictionary font.

from visdom import Visdom
"""
Lists of indices representing connections between specific facial landmarks.
"""
class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env='main'):
        self.viz = Visdom()
        self.env = env
        self.plots = {}
    def plot(self, var_name, xlabel, ylabel, legend, title, x, y):
        # print(x)
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(
                X= np.array(x) if len(x) == 0 else np.column_stack((x)), 
                Y= np.array(y) if len(y) == 0 else np.column_stack((y)),
                env=self.env, 
                opts=dict(
                    legend=legend,
                    title=title,
                    xlabel=xlabel,
                    ylabel=ylabel
                )
            )
        else:
            self.viz.line(
                X=np.array(x) if len(x) == 0 else np.column_stack((x)), 
                Y=np.array(y) if len(y) == 0 else np.column_stack((y)), 
                env=self.env, 
                win=self.plots[var_name], 
                update ='append',
                opts=dict(
                        legend=legend,
                        title=title,
                        xlabel=xlabel,
                        ylabel=ylabel
                    )
            )

class gradPlotter():
    def __init__(self, env_name, win_name):
        self.visPlotter = VisdomLinePlotter(env=env_name)
        self.win_name = win_name

    def appendData(self, named_parameters, iter_no):
        X, Y, leg = [], [], []
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                Y.append(p.grad.abs().mean().item())
                X.append(iter_no)
                leg.append(n)
        self.visPlotter.plot(self.win_name, 'iterations', 'GRADS', leg, self.win_name, X, Y)

def plot_grad_flow(named_parameters, vis):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
   
    win = vis.matplot(plt, opts={'resizable': True})
    

def write_video_cv(frames, speech, fs, path, fname, fps):   ##  This line defines a function named write_video_cv that takes six arguments: frames, speech, fs, path, fname, and fps.
    print(frames.shape, (True if len(frames.shape) == 4 else False))
    out = cv2.VideoWriter(os.path.join(path, fname), cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (frames.shape[1], frames.shape[2]), (True if len(frames.shape) == 4 else False))  ## This line prints the path and filename where the video will be saved. It uses os.path.join() to concatenate the directory path (path) and the filename (fname).
    """
   This line initializes a VideoWriter object from OpenCV (cv2). It creates a video file at the specified path with the given filename (fname). The video codec used is specified as mp4v. The frame rate (fps) and frame size are also specified based on the dimensions of the input frames (frames). If the frames have a 4th dimension (indicating color channels), it sets isColor to True, indicating that the video is in color. 
    """
    if out.isOpened():  ## This line checks if the video writer object is successfully opened.
        for i in range(frames.shape[0]):   ## This line iterates over each frame in the frames array.
            out.write(frames[i, ...])  ##  This line writes the current frame to the video file.
    out.release()  ## This line releases the video writer object, indicating that writing is complete.


    #librosa no longer has output from v0.8 lrpAdd
    #librosa.output.write_wav(os.path.join(path, fname+'.wav'), speech, fs)     ## This line writes the audio data (speech) to a WAV file at the specified path with the filename fname and the extension .wav.
    sf.write(os.path.join(path, fname+'.wav').format(chr(int(i/50)+65)), speech, fs)


    cmd = 'ffmpeg -i '+os.path.join(path, fname)+' -i '+os.path.join(path, fname)+'.wav -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0  '+os.path.join(path, fname)+'_.mp4'  ## This line constructs a command string for executing FFmpeg to mux (combine) the video file (fname) and the audio file (fname.wav) into a single MP4 file.
    subprocess.call(cmd, shell=True)   ### This line calls the FFmpeg command using the subprocess module to execute it in the shell.
    print('Muxing Done')

    # cmd = 'ffmpeg -i ' + os.path.join(path, fname) +'_.mp4 -filter:v fps=fps=120 ' + os.path.join(path, fname) + '_inter.mp4'
    # subprocess.call(cmd, shell=True) 
    # print('Muxing Done')

    os.remove(os.path.join(path, fname))  ## Removes any existing video file with the same name in the specified path to avoid conflicts.

    os.remove(os.path.join(path, fname+'.wav'))  ## These lines remove the original video file and audio file, as they have been muxed into a single file.

def plotGrads(data, lab, figsize=(10, 10)):
    plt.figure(figsize=figsize)
    plt.plot(data['G_SE'], 'r', label='spch_encoder')
    plt.plot(data['G_IE'], 'b', label='img_encoder')
    plt.plot(data['G_D'], 'g', label='decoder')
    plt.xlabel('Epoch')
    plt.ylabel('Norm')
    plt.grid(True)
    plt.legend()
    plt.savefig(lab, dpi = 300, bbox_inches='tight')
    plt.clf()
    plt.close()

def plotLosses(data, lab, figsize=(10, 10)):
    plt.figure(figsize=figsize)
    plt.plot(data['train'], 'r', label='train')
    plt.plot(data['val'], 'b', label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(lab, dpi = 300, bbox_inches='tight')
    plt.clf()
    plt.close()

def plotLossesRecent(data, lab, figsize=(10, 10)):
    if len(data['train']) < 10: 
        return
    plt.figure(figsize=figsize)
    plt.plot(data['train'][-10:], 'r', label='train')
    plt.plot(data['val'][-10:], 'b', label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(lab, dpi = 300, bbox_inches='tight')
    plt.clf()
    plt.close()

def plotFilters(data, divNum, lab):
    numItems = data.shape[-1]
    plt.figure(figsize=(divNum[0], divNum[1]))
    gs1 = gridspec.GridSpec(divNum[0], divNum[1])
    gs1.update(wspace=0.025, hspace=0.05)
    cnt = 0
    
    for i in range(numItems):
        ax1 = plt.subplot(gs1[i])
        ax1.imshow(data[:, :, i], cmap='gray')
        plt.axis('on')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect('equal')
    
    plt.savefig(lab, dpi = 300, bbox_inches='tight')
    plt.clf()
    plt.close()

def normImg(I):
    I = I - np.min(I)
    return (255*I/np.max(I)).astype(np.uint8)

def plotPaper(lab, images, dim = (10, 6), titles = None):
    n_images = len(images)
    # if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure(figsize=dim)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    for n, image in enumerate(images):
        a = fig.add_subplot(dim[1], dim[0], n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image[:, :, [2, 1, 0]])
        a.axes.get_xaxis().set_ticks([])
        a.axes.get_yaxis().set_ticks([])
        if n == dim[0]*dim[1]:
            break
    plt.savefig(lab, bbox_inches = 'tight',
        pad_inches = 0)
    plt.clf()
    plt.close()


def plotAligned(lab, images, cols = 3, titles = None):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    n_images = len(images)
    # if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure(figsize=(3, 1))
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    for n, image in enumerate(images):
        a = fig.add_subplot(1, cols, n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image[:, :, [2, 1, 0]])
        a.axes.get_xaxis().set_ticks([])
        a.axes.get_yaxis().set_ticks([])
        if n == 3:
            break
    plt.savefig(lab, bbox_inches = 'tight',
        pad_inches = 0)
    plt.clf()
    plt.close()

def plotMultiImages(data, divNum):
    num = data.shape[-1]
    r = data.shape[0]
    c = data.shape[1]

    I = np.zeros((r*divNum[0], c*divNum[1]))
    # print(I.shape)

    for i in range(num):
        cur_r = (i%divNum[0])*r
        cur_c = (i//divNum[0])*c
        img =normImg(data[:, :, i])
        # print(np.mean(img), np.std(img), np.min(img), np.max(img))
        I[cur_r:cur_r+r, cur_c:cur_c+c] = img

    return I

def easy_show_FLM(data, lmarks, lab, figsize=(10, 10)):
    plt.figure(figsize=figsize)
    plt.imshow(data)
    plt.plot(lmarks[:, 0], lmarks[:, 1], 'r*')
    plt.savefig(lab, dpi = 300, bbox_inches='tight')
    plt.clf()
    plt.close()
    """
    This function is similar to easy_show, but it also overlays facial landmarks (lmarks) on the image before saving it. After displaying the image with plt.imshow(data), it plots the facial landmarks as red asterisks using plt.plot(lmarks[:, 0], lmarks[:, 1], 'r*'). Then, it saves the figure with the provided label and closes it as before.
    """
def easy_show(data, lab, cmap='jet', figsize=(10, 10)):
    plt.figure(figsize=figsize)
    plt.imshow(data, cmap=cmap)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.gca().get_xaxis().set_ticks([])
    plt.gca().get_yaxis().set_ticks([])
    plt.savefig(lab, dpi = 300, bbox_inches='tight')
    plt.clf()
    plt.close()  
    """
    This function is for displaying an image (data) and saving it with a label (lab). It uses Matplotlib to create a figure with a specified size (figsize), then displays the image using plt.imshow(). It saves the figure as an image file with the provided label using plt.savefig(). Finally, it clears the current figure and closes it using plt.clf() and plt.close().

    """

def write_video(frames, sound, fs, path, fname, fps, cmap='jet'):
    try:
        os.remove(os.path.join(path, fname+'.mp4'))
        os.remove(os.path.join(path, fname+'.wav'))
        os.remove(os.path.join(path, fname+'_ws.mp4'))
    except:
        print ('Exp')

    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')  ### Defines metadata for the video.

    writer = FFMpegWriter(fps=fps, metadata=metadata)  ## Creates an instance of the FFMpeg writer with the specified frames per second (fps) and metadata.

    fig = plt.figure(figsize=(10, 10))
    l = plt.imshow(frames[0], cmap=cmap)


    #librosa no longer has output from v0.8 lrpAdd
    #librosa.output.write_wav(os.path.join(path, fname+'.wav'), sound, fs)
    sf.write(os.path.join(path, fname+'.wav').format(chr(int(i/50)+65)), sound, fs)


    with writer.saving(fig, os.path.join(path, fname+'.mp4'), 150):
        # plt.gca().invert_yaxis()
        plt.axis('off')
        for i in tqdm(range(frames.shape[0])):
            l.set_data(frames[i])
            cnt = 0
            writer.grab_frame()

    cmd = 'ffmpeg -i '+os.path.join(path, fname)+'.mp4 -i '+os.path.join(path, fname)+'.wav -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0  '+os.path.join(path, fname)+'_.mp4'
    subprocess.call(cmd, shell=True) 
    print('Muxing Done')

    os.remove(os.path.join(path, fname+'.mp4'))
    os.remove(os.path.join(path, fname+'.wav'))

    plt.clf()
    plt.close()

def write_video_FLM(frames, sound, fs, path, fname, xLim, yLim, fps=29.97):
    try:
        os.remove(os.path.join(path, fname+'.mp4'))
        os.remove(os.path.join(path, fname+'.wav'))
        os.remove(os.path.join(path, fname+'_ws.mp4'))
    except:
        print ('Exp')

    if len(frames.shape) < 3:
        frames = np.reshape(frames, (frames.shape[0], int(frames.shape[1]/2), 2))
    # print frames.shape

    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')
    writer = FFMpegWriter(fps=fps, metadata=metadata)

    fig = plt.figure(figsize=(10, 10))
    l, = plt.plot([], [], 'ko', ms=4)


    plt.xlim(xLim)
    plt.ylim(yLim)

    #librosa no longer has output from v0.8 lrpAdd
    #librosa.output.write_wav(os.path.join(path, fname+'.wav'), sound, fs)   ### Writes the audio data (sound) to a WAV file with the specified filename in the specified path.
    sf.write(os.path.join(path, fname+'.wav').format(chr(int(i/50)+65)), sound, fs)



    if frames.shape[1] == 20:
        lookup = [[x[0] - 48, x[1] - 48] for x in Mouth]
        # print lookup
    else:
        lookup = faceLmarkLookup

    lines = [plt.plot([], [], 'k')[0] for _ in range(3*len(lookup))]

    with writer.saving(fig, os.path.join(path, fname+'.mp4'), 150):
        plt.gca().invert_yaxis()   ## Inverts the y-axis to match the usual orientation of images.
        for i in tqdm(range(frames.shape[0])):
            l.set_data(frames[i,:,0], frames[i,:,1])
            cnt = 0
            for refpts in lookup:
                lines[cnt].set_data([frames[i,refpts[1], 0], frames[i,refpts[0], 0]], [frames[i, refpts[1], 1], frames[i,refpts[0], 1]])
                cnt+=1
            writer.grab_frame()

    cmd = 'ffmpeg -i '+os.path.join(path, fname)+'.mp4 -i '+os.path.join(path, fname)+'.wav -c:v copy -c:a aac -strict experimental '+os.path.join(path, fname)+'_.mp4'
    subprocess.call(cmd, shell=True) 
    print('Muxing Done')

    os.remove(os.path.join(path, fname+'.mp4'))
    os.remove(os.path.join(path, fname+'.wav'))
