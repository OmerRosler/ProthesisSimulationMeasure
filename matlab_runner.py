import os
import cv2
from contextlib import contextmanager
from parse import parse

import matlab.engine

from extract_frames_from_video import extract_frame_at_time, video_capture_context
from compare_images import compare_contours

default_values = {'SigmaX':0.13,
    'SigmaY':0.13,
    'X0':7.0,
    'Y0':7.0,
    'DFR':60.0,
    'EAR':10.0,
    'PER':2.0,
    'PFO':10.0,
    'PFD':10.0,
    'Polarity':1,
    'ElectrodeArraySize':3,
    'ElectrodeArrayStructure':1
}
def file_iterator(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            yield file_path

dir_name_format_string = "{video_name}_{SigmaX:.3f}_{SigmaY:.3f}_{X0:.3f}_{Y0:.3f}_{DFR:.3f}_{EAR:.3f}_{PER:.3f}_{PFO:.3f}_{PFD:.3f}_{Polarity:.3f}_{ElectrodeArraySize:.3f}_{ElectrodeArrayStructure:.3f}"

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
    # Call a MATLAB function
    video_name = eng.main_function(videofile, SigmaX, SigmaY, X0, Y0, DFR, EAR, PER, PFO,PFD, Polarity, ElectrodeArraySize, ElectrodeArrayStructure)

    print(f'The video created is {video_name}')


def run_simulation_if_new_params(eng, videofile, **kwargs):
    video_name,_ = os.path.splitext(videofile)
    print(video_name)
    directory = dir_name_format_string.format(video_name = video_name, **kwargs)
    if not os.path.isdir(directory):
        run_matlab(eng, videofile, **kwargs)
    return directory

def measure_contour_detection(eng, original_videofile, **kwargs):
    directory = run_simulation_if_new_params(eng, original_videofile,**kwargs)
    # Get first frame of the original video and save as image
    with video_capture_context(original_videofile) as cap:
        frame_id, first_frame = extract_frame_at_time(cap,0)
        original_frame_filename = os.path.join(directory, f'orig_frame_{frame_id:04d}.png')
        cv2.imwrite(original_frame_filename, first_frame)
    # Get last frame of the synthesized video and save as image
    new_video = os.path.join(directory, 'ProstheticVideo.mp4')
    if not os.path.isfile(new_video):
        raise ValueError("No video generated")
    with video_capture_context(new_video) as cap:
        # The videos created stop after the movement, where we extract the frame
        last_frame_id, last_frame = extract_frame_at_time(cap,-1)
        analyzed_frame_filename = os.path.join(directory, f'orig_frame_{last_frame_id:04d}.png')
        cv2.imwrite(analyzed_frame_filename, last_frame)

    # Compare them using the chosen method
    return compare_contours(original_frame_filename, analyzed_frame_filename)

eng = matlab.engine.start_matlab()
print(measure_contour_detection(eng, 'HeadMovementEllipse.mp4', **default_values))
print(measure_contour_detection(eng, 'HeadMovementRect.mp4', **default_values))
print(measure_contour_detection(eng, 'HeadMovementTriangle.mp4', **default_values))
# Stop the MATLAB engine
eng.quit()