import argparse  ## This imports the argparse module, which is a standard Python library used for parsing command-line arguments.
import os  ## This imports the os module, which provides a portable way of using operating system dependent functionality.
import subprocess  ## This imports the subprocess module, which allows you to spawn new processes, connect to their input/output/error pipes, and obtain their return codes.

if __name__ == '__main__':  ## This imports the subprocess module, which allows you to spawn new processes, connect to their input/output/error pipes, and obtain their return codes.
    parser = argparse.ArgumentParser(description=__doc__) ## This creates an ArgumentParser object, which is used to parse the command-line arguments. The description parameter sets the description of the program using the module's docstring (__doc__).
    parser.add_argument("-i", "--input-folder", type=str, help='Path to folder that contains video files') ##  This adds a command-line argument to the parser. -i and --input-folder are the short and long versions of the argument flag respectively. type=str specifies that the argument should be interpreted as a string. The help parameter provides a brief description of what the argument does.
    parser.add_argument("-fps", type=float, help='Target FPS', default=25.0)   ##  This adds another command-line argument to the parser. -fps is the flag for this argument. type=float specifies that the argument should be interpreted as a floating-point number. The default parameter sets a default value for the argument if it is not provided.

    parser.add_argument("-o", "--output-folder", type=str, help='Path to output folder')  ##  This adds another command-line argument to the parser. -o and --output-folder are the short and long versions of the argument flag respectively. type=str specifies that the argument should be interpreted as a string. The help parameter provides a brief description of what the argument does
    args = parser.parse_args()   ##  This parses the command-line arguments using the ArgumentParser object created earlier and stores the parsed arguments in the args variable.

    os.makedirs(args.output_folder, exist_ok=True)  ##  This creates the output folder specified by the user if it does not already exist. The exist_ok=True parameter ensures that the function does not raise an error if the folder already exists

    fileList = []  ##  This initializes an empty list called fileList to store the paths of video files.
    for root, dirnames, filenames in os.walk(args.input_folder):  ## This iterates over the directory tree rooted at the input_folder provided by the user using the os.walk() function. It returns the root directory, the directories within it, and the filenames within it.
        for filename in filenames:  ##  This iterates over the filenames found in the current directory being walked.
            if os.path.splitext(filename)[1] == '.mp4' or os.path.splitext(filename)[1] == '.mpg' or os.path.splitext(filename)[1] == '.mov' or os.path.splitext(filename)[1] == '.flv':  ## This checks if the current filename has one of the specified video file extensions (mp4, mpg, mov, flv). If so, it appends the full path of the file to the fileList.
                fileList.append(os.path.join(root, filename))

    for file in fileList:  ## This iterates over the list of video files stored in fileList.
        subprocess.run("ffmpeg -i {} -r 25 -y {}".format(file, os.path.splitext(file.replace(args.input_folder, args.output_folder))[0]+".mp4"), shell=True)  ## This line invokes the ffmpeg command using subprocess to convert each video file to a target FPS of 25 and saves the converted file in the output folder specified by the user

