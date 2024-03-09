import argparse  ## Imports the argparse module, which is used for parsing command-line arguments
import os  ## Imports the os module, which provides a portable way of using operating system-dependent functionality
import shutil  ## Imports the shutil module, which offers a higher-level interface for file operations than os.

import h5py  ## Imports the h5py module, which is used for reading and writing HDF5 files.
# import librosa
import numpy as np  ## Imports the numpy module and aliases it as np. NumPy is a library for numerical computations in Python.
from tqdm import tqdm   ## Imports the tqdm function from the tqdm module. tqdm is used to create progress bars for loops
from scipy.signal import decimate   ## Imports the decimate function from the scipy.signal module. It is used for signal decimation.
from scipy import interpolate  ## Imports the interpolate module from SciPy, which provides functions for interpolating between data points.
from scipy import signal   ## Imports the signal module from SciPy, which provides signal processing functions.
from scipy.signal import butter, lfilter, freqz, wiener   ## Imports specific functions (butter, lfilter, freqz, wiener) from the scipy.signal module.
from scipy.io import wavfile  ## Imports the wavfile function from the scipy.io module. It is used for reading WAV files
import multiprocessing as mp  ## Imports the multiprocessing module and aliases it as mp. It is used for parallel processing.
import utils  ## Imports a module named utils. This likely contains utility functions used in the script.
from facealigner import FaceAligner, crop_image  ## Imports the FaceAligner class and the crop_image function from a module named facealigner.
from helper import shape_to_np  ## Imports the shape_to_np function from a module named helper.
import cv2  ## Imports the OpenCV library for computer vision tasks.
import dlib  ## Imports the Dlib library, often used for facial recognition tasks.
import math  ## Imports the math module, which provides mathematical functions.
from multiprocessing import Process, Queue, Manager  ## Imports specific components (Process, Queue, Manager) from the multiprocessing module.
import subprocess  ## Imports the subprocess module, which allows you to spawn new processes, connect to their input/output/error pipes, and obtain their return codes.
import random as rn ## Imports the subprocess module, which allows you to spawn new processes, connect to their input/output/error pipes, and obtain their return codes.
from random import randint, shuffle  ## Imports specific functions (randint, shuffle) from the random module.
from collections import deque  ## Imports the deque class from the collections module. deque is a double-ended queue.

from scipy.interpolate import CubicSpline   ##Imports the CubicSpline function from the scipy.interpolate module.CubicSpline is used to perform cubic spline interpolation.
#-----------------------------------------#
#           Reproducible results          #
#-----------------------------------------#
os.environ['PYTHONHASHSEED'] = '123'##Sets the environment variable PYTHONHASHSEED to the string '123'. This is used to control the hashing algorithm's seed value in Python, ensuring reproducible hash values across different executions.
np.random.seed(123)
rn.seed(123) ## Seeds the Python built-in random module's random number generator with the integer 123, ensuring reproducibility of random number generation.
# torch.manual_seed(999)
#-----------------------------------------#
#----------------PARSER------------------#
parser = argparse.ArgumentParser(description=__doc__) # this line initializes the argumentparser object from 'argparse' module. The 'description' parameter is set to '__doc__',which typically contains the module's docstring. This description is displayed when the user asks for help using '__help' option .Specifies an argument named input-folder (short form -i, long form --input-folder). It expects a string value and represents the path to the folder that contains video files.
parser.add_argument("-i", "--input-folder", type=str, help='Path to folder that contains video files')  ## Adds command-line arguments to the parser. Each add_argument call specifies a particular argument that the program can accept. Here's what each argument means:
parser.add_argument("-p", "--pred-path", type=str, help="Predictor Path", default='data/dlib_data/shape_predictor_68_face_landmarks.dat')
parser.add_argument("-ti", "--temp-img", type=str, help="Template face image path", default='data_prep/template_face.jpg')  ## Specifies an argument named temp-img (short form -ti, long form --temp-img). It expects a string value and represents the path to the template face image used for alignment. The default value is 'data_prep/.jpg'.
parser.add_argument("-fs", "--fs", type=int, help='sampling rate', default=8000) ## Specitemplate_facefies an argument named fs (short form -fs, long form --fs). It expects an integer value and represents the sampling rate. The default value is 8000.
parser.add_argument("-nw", "--nw", type=int, help='num workers', default=1)  ## Specifies an argument named nw (short form -nw, long form --nw). It expects an integer value and represents the number of workers. The default value is 1.
parser.add_argument("-d", "--debug", type=bool, help='Writes videos for debug purposes', default=False) ##  Specifies an argument named debug (short form -d, long form --debug). It expects a boolean value and indicates whether to write videos for debugging purposes. The default value is False.
parser.add_argument("-o", "--out-path", type=str, help='Output path')  ## Specifies an argument named out-path (short form -o, long form --out-path). It expects a string value and represents the output path.
args = parser.parse_args()  ##  Parses the command-line arguments provided by the user and stores them in the args variable. This line effectively captures the user-provided arguments for further use within the script.

in_path = args.input_folder # this line assigns the value of  the 'input folder' argument passed through the command line to the variable 'in_path'
fs = args.fs
out_path = args.out_path
TI_path = args.temp_img
num_processes = args.nw
print("num_processes: ", num_processes)
predictor_path = args.pred_path

if not os.path.exists(out_path):   ## This condition checks if the specified output path (out_path) does not exist.If the path does not exist, it means that there is no directory at the specified location, so the script needs to create one
    os.makedirs(out_path)   ## If the output path does not exist, this line creates the directory (and any necessary parent directories) specified by out_path.  os.makedirs() is used to recursively create directories.

else:
    shutil.rmtree(out_path)  ## This line removes the existing directory specified by out_path and all its contents recursively.
    os.mkdir(out_path)   ## shutil.rmtree() is used to remove a directory and its contents.   After removing the existing directory, this line creates a new directory at the same location.

class TemplateProcessor():
    def __init__(self, path): # defines the constructure method for templateprocessor class .it takes one argument 'path'
        template_I = cv2.imread(path)    ##  Reads an image file specified by the path argument using OpenCV's imread function and stores it in the variable template_I.

        detector = dlib.get_frontal_face_detector()  ## Initializes a face detector using Dlib's get_frontal_face_detector function and stores it in the variable detector. This detector is capable of detecting faces in images.

        predictor = dlib.shape_predictor(predictor_path)   ## tializes a shape predictor using Dlib's shape_predictor function and the predictor_path variable (which likely contains the path to a pre-trained shape predictor model). This predictor is used to detect facial landmarks in images.
        fa = FaceAligner(predictor, desiredLeftEye=(0.25, 0.25), desiredFaceWidth=128)  ## Initializes a FaceAligner object (assuming it's a custom class) with the shape predictor and desired parameters for face alignment, such as the desired position of the left eye (desiredLeftEye) and the desired width of the aligned face (desiredFaceWidth).
        gray = cv2.cvtColor(template_I, cv2.COLOR_BGR2GRAY)   ## Converts the input image (template_I) to grayscale using OpenCV's cvtColor function. This is often done before performing facial landmark detection to simplify processing.

        dets = detector(template_I, 1)   ## Detects faces in the grayscale image (gray) using the previously initialized face detector (detector). The 1 argument likely indicates that the detector should only return a single face detection result.
        for k, d in enumerate(dets):   ##  Iterates over the detected face regions (d) returned by the face detector.
            shape = predictor(gray, d)  ##  Detects facial landmarks (such as the positions of the eyes, nose, and mouth) within the detected face region (d) using the shape predictor (predictor) and the grayscale image (gray). The detected landmarks are stored in the variable shape.
    
        template_I, scale = fa.align(template_I, gray, d, shape, None)  ## Aligns the original input image (template_I) to the detected face region (d) and the detected facial landmarks (shape) using the align method of the FaceAligner object (fa). This likely performs geometric transformations to normalize the face position and scale.

        gray = cv2.cvtColor(template_I, cv2.COLOR_BGR2GRAY) #Converts the input image (template_I) to grayscale using OpenCV's cvtColor function. This is often done             
                                                                  before performing facial landmark detection to simplify processing
        dets = detector(template_I, 1)
        for k, d in enumerate(dets):     #Detects faces in the grayscale image (gray) using the previously initialized face detector (detector). The 1 argument likely   
            shape = predictor(gray, d)        # indicates that the detector should only return a single face detection result

        template_I = cv2.cvtColor(template_I, cv2.COLOR_BGR2RGB)  ## Converts the aligned image (template_I) to grayscale again. This step may be redundant if the alignment process preserves the grayscale information.
        utils.easy_show(template_I, 'template_face.png')

        scl = 0.6

        shape = shape_to_np(shape)
        shape -= np.tile(np.min(shape, axis=0), [68, 1])
        shape = ((128*(shape/np.tile(np.max(shape, axis=0), [68, 1])))*scl) + 128*((1-scl)/2)

        self.points = shape
        """
        Face Detection and Landmark Extraction:

It uses the detector to detect faces in the template_I image.
For each detected face (d), it extracts facial landmarks using the predictor.
These facial landmarks are stored in the variable shape.
Displaying the Template Image:

It converts the template image (template_I) from BGR to RGB color space using cv2.cvtColor.
It displays the template image using a utility function utils.easy_show.
Scaling and Normalizing Facial Landmarks:

It defines a scaling factor scl (0.6).
It converts the facial landmarks (shape) to a NumPy array using shape_to_np.
It subtracts the minimum values of each dimension from the facial landmarks.
It scales the facial landmarks to fit within a range of [128 * (1 - scl), 128 * scl] along each dimension.
It shifts the scaled landmarks to be centered around the origin (128, 128).
Storing Landmarks:

It assigns the scaled and normalized landmarks to self.points, making them accessible through the getShape method.
        """
    def getShape(self):
        return self.points


class Extractor():
    def __init__(self, in_path, mean_shape): #Defines the constructor method (__init__) for the Extractor class. It takes three parameters: in_path, which represents # the input path containing video files, and mean_shape, which seems to represent some mean shape data.
        self.dataQueue = Manager().Queue()   ##Initializes a shared `Queue` object using `Manager()`. This queue is likely intended to be used for communication between processes.   
        self.fileList = []  ##  Initializes an empty list named `fileList`. This list will be used to store tuples containing file information (directory path and filename).
        self.mean_shape = mean_shape   ## Assigns the `mean_shape` parameter to the instance variable `self.mean_shape`.
        for root, dirnames, filenames in os.walk(in_path):  ## Iterates over the directory tree rooted at `in_path`. For each directory visited, it yields a 3-tuple containing the directory path, the list of subdirectories, and the list of files.
            for filename in filenames:  ##  Iterates over the list of filenames in the current directory.
                if os.path.splitext(filename)[1] == '.mp4' or os.path.splitext(filename)[1] == '.mpg' or os.path.splitext(filename)[1] == '.mov' or os.path.splitext(filename)[1] == '.flv':
                    self.fileList.append((root, filename))
                 """
                 . `if os.path.splitext(filename)[1] == '.mp4' or os.path.splitext(filename)[1] == '.mpg' or os.path.splitext(filename)[1] == '.mov' or os.path.splitext(filename)[1] == '.flv':`: Checks if the file extension of the current filename matches any of the video file extensions ('.mp4', '.mpg', '.mov', '.flv'). `os.path.splitext()` splits the filename into its base name and extension, and `[1]` selects the extension part.## If the file is a video file, it appends a tuple containing the directory path (`root`) and the filename to the `fileList`.
                 """
        self.fileList = self.fileList[:]    ##This line is redundant and doesn't have any effect. It creates a shallow copy of `self.fileList`, but assigns it back to `self.fileList` without any change.
        print(self.fileList)  ## Prints the list of tuples containing file information.
        # exit()   ## This is a commented-out line that, if uncommented, would exit the program immediately after printing the file list.
   """
   In summary, the `Extractor` class initializes a shared queue and constructs a list of video files (`fileList`) found within a specified directory (`in_path`). Each video file is represented as a tuple containing its directory path and filename.
   """     
    def processSample(self, process_id): #Defines a method named processSample which takes process_id as an argument. This method is responsible for processing a subset of samples identified by process_id 
            import librosa                       ## Imports the librosa library, which is commonly used for audio and music signal processing tasks.
            n = len(self.fileList)  ## Computes the total number of files in fileList.
            increment = n // num_processes  ## Computes the increment value for splitting the samples among different processes.
            if process_id == num_processes-1:   ##  Checks if the current process is the last one.
                sampleList = list(range(process_id*increment, n))  ## If it's the last process, assigns the remaining samples to the current process.
            else:   ## If it's not the last process:
                sampleList = list(range(process_id*increment, (process_id+1)*increment))  ## Assigns a subset of samples to the current process.
            print(process_id, len(sampleList), n,  process_id*increment, (process_id+1)*increment)   ### Prints information about the current process, such as process ID, number of samples assigned to the process, total number of files, and the range of sample indices assigned to the process.

            # exit()

            detector = dlib.get_frontal_face_detector()  ## Initializes a face detector using the Dlib library.
            predictor = dlib.shape_predictor(predictor_path)   ## Initializes a facial landmark predictor using the Dlib library.
            desired_dim = 128   ## Sets the desired dimension for the aligned face images.
            fa = FaceAligner(predictor, desiredLeftEye=(0.25, 0.25), desiredFaceWidth=desired_dim)  ## Initializes a FaceAligner object for aligning faces using the given predictor and desired parameters.

            shuffle(sampleList)  ## Shuffles the list of sample indices to introduce randomness in processing.

            for j in tqdm(sampleList):  ## Iterates over the shuffled sample indices, displaying a progress bar using tqdm.
                print(j)
                root, filename = self.fileList[j]  ## Retrieves the directory path and filename corresponding to the current sample index.
                
                y, sr = librosa.load(os.path.join(root, filename), sr=fs)  ## Loads the audio file using librosa.load.

                try:
                    y = y-np.mean(y)  ## Normalizes the audio signal by subtracting its mean.
                    speech = y/np.max(np.abs(y)) ## Normalizes the audio signal by dividing it by its maximum absolute value.
                    if np.isnan(speech).any():
                        print('NaN encountered! Skipping file...')
                        continue
                except:
                    print('Exception! Skipping file...')
                    continue

                frame_list = []   ## Initializes lists to store processed frames and facial landmarks.
                lmarks_list = []
                try:
                    cap = cv2.VideoCapture(os.path.join(root, filename))  ## Opens the video file using OpenCV's VideoCapture class.
                    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) ## Retrieves the total number of frames in the video.
                    print("Length: ", length)
                    dset = h5py.File(os.path.join(out_path, os.path.splitext(filename)[0]) + '.hdf5', 'w')  ## Opens an HDF5 file for writing processed data.
                    frame_cnt = 0   
                    scale = None
                    video = []
                    tmp_lmarks = []
                    for frame_cnt in range(length):   ## Iterates over each frame in the video.
                        ret, frame = cap.read()  ##  Reads a frame from the video.
                        if frame is None:
                            break
                        video.append(frame)   ## Appends the frame to the video list.
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Converts the frame to grayscale.

                        dets = detector(frame, 1)  ## Detects faces in the frame using the face detector.
                        for k, d in enumerate(dets):  ## Iterates over the detected faces.
                            tmp_lmarks.append(shape_to_np(predictor(gray, d)))    ## Extracts facial landmarks for each detected face and appends them to a temporary list.

                    lmarks = np.array(tmp_lmarks)  ## Converts the temporary list of facial landmarks into a NumPy array.
                    
                    if args.mode == 1:  ## Checks if the alignment mode is set to 1.
                        fa.get_tform(video[0], lmarks[0, ...], self.mean_shape, scale)  ## Computes the transformation to align the first frame using the mean shape.
                    
                    for frame_cnt in range(length):  ##  Iterates over each frame in the video again.
                        # frame, scale = fa.align_box(video[frame_cnt], lmarks[frame_cnt, ...], self.mean_shape, scale)
                        # frame, scale = fa.align_three_points(frame, np.average(np.array(buffer), axis=0, weights=[x/sum(list(range(buffersize))) for x in range(buffersize)]), self.mean_shape, scale)
                        if args.mode == 1:  ## Checks if the alignment mode is set to 1.
                            frame, scale = fa.apply_tform(video[frame_cnt])  ## Applies the computed transformation to align the current frame.
                        else:
                            frame, scale = fa.align_three_points(video[frame_cnt], lmarks[frame_cnt, ...], self.mean_shape, scale)  ## If mode is 0, aligns each frame by identifying and aligning three characteristic points on the face using the align_three_points method.


                       
                        frame_list.append(frame)  ## Appends the aligned frame to the frame_list.
                        
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  ## Converts the aligned frame to grayscale.
                        dets = detector(frame, 1)  ## Detects faces in the aligned frame.
                        for k, d in enumerate(dets):  ## Iterates over the detected faces.
                            shape = shape_to_np(predictor(gray, d))   ## xtracts facial landmarks for each detected face using the shape predictor.
                        lmarks_list.append(shape)       ## Appends the extracted facial landmarks to the lmarks_list.                 

                        frame_cnt+=1   ## Increment the frame counter.
                    
                    dset.create_dataset('speech', data=speech) ##  Creates a dataset named 'speech' in the HDF5 file and stores the speech data.
                    dset.create_dataset('video', data=np.array(frame_list))  ##  Creates a dataset named 'video' in the HDF5 file and stores the aligned frames.
                    dset.create_dataset('lmarks', data=np.array(lmarks_list))  ## Creates a dataset named 'lmarks' in the HDF5 file and stores the extracted facial landmarks.
                    self.dataQueue.put(('dummy', None))  ## Puts a tuple ('dummy', None) into the data queue. This could be a signal to indicate that a sample has been processed
                except:  ## If an exception occurs during processing, it catches the exception.
                    os.remove(os.path.join(out_path, os.path.splitext(filename)[0]) + '.hdf5')       ## Deletes the HDF5 file associated with the current sample if an exception occurs.     
                    print('Exception! Deleting file...', os.path.join(out_path, os.path.splitext(filename)[0]) + '.hdf5')
                    continue
                # utils.write_video_cv(np.array(frame_list), speech, 8000, '', '{}_test.mp5'.format(j), 25.0)
                # if j > 9:
                #     exit()
                if args.debug: ##  Checks if the debug mode is enabled.
                    utils.write_video_cv(np.array(frame_list), speech, 8000, out_path, os.path.join(os.path.splitext(filename)[0] + '.mp4'), 25.0) ## Writes the processed frames and speech data to a video file if debug mode is enabled.

            self.dataQueue.put(('end', process_id))  #his line puts a tuple containing a dummy value ('dummy') and None into a multiprocessing queue (self.dataQueue). Signals the end of processing for the current process by putting a tuple ('end', process_id) into the data queue.
            print('Thread Ended #', process_id)           ##Prints a message indicating the end of processing for the current process. This is typically used for synchronization or signaling between different processes

    def writeToFile(self):
        cnt = 0  ## initializes a counter variable cnt to keep track of the number of processed files.
        threadStatus = [0] * num_processes  ## Initializes a list threadStatus with num_processes elements, each representing the status of a processing thread. Here, 0 indicates that the thread is not yet finished processing.
        pbar = tqdm(total = len(self.fileList))  ##  Initializes a progress bar (pbar) using the tqdm library with the total number of files in self.fileList as the total count.
        while True:  ## Starts an infinite loop.
            if all(threadStatus):  ## Checks if all elements in threadStatus are True, indicating that all processing threads have finished their work.
                break  ## Breaks out of the loop if all threads have finished.
            data = self.dataQueue.get()   ## Retrieves data from the dataQueue. This is a blocking operation, meaning it waits until data is available in the queue
            if isinstance(data[0], str) and data[0] == 'end':   ## Checks if the received data is a tuple where the first element is a string and equal to 'end'. This indicates the end of processing for a particular thread
                print('End ', data[1])  ## Prints a message indicating the end of processing for the thread identified by data[1].
                threadStatus[data[1]] = 1  ## Updates the status of the thread identified by data[1] to 1, indicating that it has finished processing.
                continue   ##Skips the rest of the loop and proceeds to the next iteration.
            # speech_dset.create_dataset(data[0]+'_'+data[1], data=data[3])
            # image_seq_dset.create_dataset(data[0]+'_'+data[1], data=data[2])
            cnt += 1  ## Increments the counter cnt to track the number of processed files.
            pbar.update(1)  ## Updates the progress bar to indicate the completion of processing for one file
            # utils.write_video(data[2], data[3], 8000, out_path, data[0]+'_'+data[1], 29.97)
        print('Main Ended.') #Prints a message indicating the end of the main processing loop
        # dset.close()
        pbar.close()  ## Closes the progress bar after processing is complete.

if __name__ == '__main__':  ## Checks if the script is being run directly (not imported as a module).
    TI_process = TemplateProcessor(TI_path) #This line creates an instance of the TemplateProcessor class, passing TI_path as an argument to its constructor
    mean_shape = TI_process.getShape() #Retrieves the mean shape from the TemplateProcessor instance using the getShape method..
    del TI_process #Deletes the TI_process object from memory to free up resources

    cnt = 0 # Initializes a variable cnt to zero.
    ext = Extractor(in_path, mean_shape) #Creates an instance of the Extractor class, initializing it with the specified in_path and mean_shape.

    if num_processes < 2:
        ext.processSample(0) #If there's only one process (if block), it directly calls the ext.processSample(0) method to process the data without parallelization.
    else:
        processes = []
        for i in range(num_processes): #creating a new Process object for each process and appending it to the processes list
            processes.append(Process(target=ext.processSample, args=(i, ))) #
        
        for i, p in enumerate(processes):
            # p.daemon = True
            p.start() #Starts each process using p.start() and prints a message indicating the start of each process.
            print('Process #', i)
        
        p = Process(target=ext.writeToFile, args=()) #Creates a new process p to execute the ext.writeToFile() method. Starts the p process.

        p.start()  ## Starts the process for writing to a file.
        p.join()  ## Waits for the writing process to finish.
        print('Main joined.') #Prints a message indicating that the main process has joined with the p process

        for i, p in enumerate(processes): #Iterates over the processes list and waits for each process to complete using p.join(), printing a message indicating the 
            p.join()                       #joining of each process.
            print('Joined #', i)
    

