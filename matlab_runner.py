import os
import itertools
import cv2
import shutil
from contextlib import contextmanager
from parse import parse
import copy

import matlab.engine

from extract_frames_from_video import extract_frame_at_time, video_capture_context
from compare_images import compare_contours, dir_name_format_string
from video_creator import generate_random_video

default_values = {'SigmaX':0.13,
    'SigmaY':0.13,
    'X0':7.0,
    'Y0':7.0,
    'DFR':60.0,
    'EAR':10.0, #Being changed
    'PER':2.0, #Being changed
    'PFO':10.0, # Being changed
    'PFD':10.0, # Needs to ask the author
    'Polarity':1,
    'ElectrodeArraySize':3,
    'ElectrodeArrayStructure':1
}

def file_iterator(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            yield file_path

#dir_name_format_string = "products/{video_name}_{SigmaX:.3f}_{SigmaY:.3f}_{X0:.3f}_{Y0:.3f}_{DFR:.3f}_{EAR:.3f}_{PER:.3f}_{PFO:.3f}_{PFD:.3f}_{Polarity:.3f}_{ElectrodeArraySize:.3f}_{ElectrodeArrayStructure:.3f}"

def validate_suite_exists(video_name, **kwargs):
    directory = dir_name_format_string.format(video_name = video_name, **kwargs)
    if not os.path.isdir(directory):
        return False
    if not os.path.isfile(os.path.join(directory,'ProstheticVideo.mp4')):
        return False
    if not os.path.isfile(os.path.join(directory,'original_video.mp4')):
        return False
    return True

def extract_parameters_from_dir_name(dirname):
    # Parse the input string
    return parse(dir_name_format_string, dirname)

def run_matlab(eng, videofile, SigmaX=0.13, SigmaY=0.13, X0=7.0, Y0=7.0, DFR=60.0, EAR=10.0, PER=2.0, PFO=10.0, PFD=10.0, Polarity=1, ElectrodeArraySize=3, ElectrodeArrayStructure=1):
    """ 
        SigmaX :   phosphene dimension in x axis
        SigmaY :   phosphene dimension in y axis
        X0 :   distance between two adjecent phosphenes in x axis
        Y0 :   distance between two adjecent phosphenes in y axis
        DFR:   display frame rate (usually 30 or 60 FPS)
        EAR:   electrode activation rate (usually 6 or 10 Hz)
        PER:   persistence duration (usually between 0 and 8 seconds)
        PFO:   perceptual fading onset (usually between 0 and 10 seconds)
        PFD:   perceptual fading duration (usually between 0.5 and 60 seconds)
        Polarity: image polarity (original(val=1) or reversed(2))
        ElectrodeArraySize :   number of electrodes (60(val=1), 256(2), 961(3), or 2916(4) electrodes)
        ElectrodeArrayStructure:   electrode array structure (square(val = 1) or hexagonal(2)).
            Currently, the hexagonal structure is pixelized when combined with temporal effects
    """
    # Clear any previous warnings
    eng.eval("lastwarn('')", nargout=0)
    try:
        engine_kwargs = {'background':True, 'nargout':1}
        # Call a MATLAB function and get the result
        video_name_future = eng.main_function(videofile, SigmaX, SigmaY, X0, Y0, DFR, EAR, PER, PFO,PFD, Polarity, ElectrodeArraySize, ElectrodeArrayStructure, **engine_kwargs)
        #print(f'The video created is {video_name}')

        
        # Check for warnings
        warning_msg = eng.eval("lastwarn")
        if warning_msg:
            print("MATLAB warning:", warning_msg)

        return video_name_future

    except matlab.engine.MatlabExecutionError as err:
        print("MATLAB execution error:", err)
        raise



def run_simulation_if_new_params(eng, video_name, videofile, **kwargs):
    if not validate_suite_exists(video_name, **kwargs):
        return run_matlab(eng, videofile, **kwargs)
    else:
        engine_kwargs = {'background':True, 'nargout':0}
        return eng.dummy_function(**engine_kwargs)

def measure_contour_detection(eng, original_videofile, **kwargs):
    base_name = os.path.basename(original_videofile)
    video_name,_ = os.path.splitext(base_name)
    directory = dir_name_format_string.format(video_name = video_name, **kwargs)
    os.makedirs(directory, exist_ok=True)
    new_video_path = os.path.join(directory, 'original_video.mp4')
    shutil.copy(original_videofile, new_video_path)
    return run_simulation_if_new_params(eng, video_name = video_name, videofile = new_video_path,**kwargs)
    #extract_frames_and_compare(original_videofile, **kwargs)
    #return future
        
def extract_frames_and_compare(original_videofile, **kwargs):
    video_name,_ = os.path.splitext(original_videofile)
    directory = dir_name_format_string.format(video_name = video_name, **kwargs)
    new_video_path = os.path.join(directory, 'original_video.mp4')
    # Get first frame of the original video and save as image
    with video_capture_context(new_video_path) as cap:
        frame_id, first_frame = extract_frame_at_time(cap,0)
        original_frame_filename = os.path.join(directory, f'first_orig_frame.png')
        cv2.imwrite(original_frame_filename, first_frame)
    # Get last frame of the synthesized video and save as image
    new_video = os.path.join(directory, 'ProstheticVideo.mp4')
    if not os.path.isfile(new_video):
        raise ValueError("No video generated")
    with video_capture_context(new_video) as cap:
        # The videos created stop after the movement, where we extract the frame
        last_frame_id, last_frame = extract_frame_at_time(cap,-1)
        analyzed_frame_filename = os.path.join(directory, f'last_analysed_frame.png')
        cv2.imwrite(analyzed_frame_filename, last_frame)

EAR_range = [10.0]
PER_range = [0.0, 2.0, 3.0, 8.0]
PFO_range = [0.5,1.0, 1.5, 2.0, 2.5, 3.0]
PFD_range = [2.0]

def generate_suite_of_simulation_parameters():
    """ These values were tested in the paper, see Table 3"""
    # For the temporal aspects, only EAR = 10 is used
    #EAR_range = [6.0, 10.0, 30.0]

    for EAR,PER,PFO,PFD in itertools.product(EAR_range, PER_range,PFO_range, PFD_range):
        values = copy.deepcopy(default_values)
        values['EAR'] = EAR
        values['PER'] = PER
        values['PFO'] = PFO
        values['PFD'] = PFD
        yield values


def list_files_os(directory):
    files = []
    for entry in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, entry)):
            files.append(os.path.join(directory, entry))
    return files
def list_files_with_prefix_os(prefix):
    files = []
    current_directory = os.getcwd()
    for entry in os.listdir(current_directory):
        if os.path.isfile(entry) and entry.startswith(prefix):
            files.append(entry)
    return files

def generate_videos(outdir = 'generated_videos/second_run'):
    for PFD in PFD_range:
        for PFO in PFO_range:
            static_start_time = PFD+PFO
            generate_random_video(static_start_time, area_ratio = 1/16, outdir=outdir)
            generate_random_video(static_start_time, area_ratio = 1/16, outdir=outdir)
            generate_random_video(static_start_time, area_ratio = 1/16, outdir=outdir)
            generate_random_video(static_start_time, area_ratio = 1/9, outdir=outdir)
            generate_random_video(static_start_time, area_ratio = 1/9, outdir=outdir)
            generate_random_video(static_start_time, area_ratio = 1/9, outdir=outdir)
        

suite = list(generate_suite_of_simulation_parameters())
#print(suite)
#generate_videos()
videos =  list_files_with_prefix_os('run2_')
engs = [None] * len(videos)
# Create a MATLAB process for each video which runs it in all 135 possible combinations
for i,_ in enumerate(videos):
    engs[i] = matlab.engine.start_matlab()
for values in suite:
    res_futures = [None] * len(videos)
    results = [None] * len(videos)
    for i,video_name in enumerate(videos):
        res_futures[i] = measure_contour_detection(engs[i], video_name, **values)
    for i,video_name in enumerate(videos):
        res_futures[i].result()
        extract_frames_and_compare(video_name, **values)

# Stop the MATLAB engines
for i,_ in enumerate(videos):
    engs[i].quit()
