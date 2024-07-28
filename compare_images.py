import os
import cv2
from itertools import groupby
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage import img_as_bool, img_as_ubyte
from parse import parse

format_witout_dir = "{video_name}_{SigmaX:.3f}_{SigmaY:.3f}_{X0:.3f}_{Y0:.3f}_{DFR:.3f}_{EAR:.3f}_{PER:.3f}_{PFO:.3f}_{PFD:.3f}_{Polarity:.3f}_{ElectrodeArraySize:.3f}_{ElectrodeArrayStructure:.3f}"
dir_name_format_string = "products/{video_name}_{SigmaX:.3f}_{SigmaY:.3f}_{X0:.3f}_{Y0:.3f}_{DFR:.3f}_{EAR:.3f}_{PER:.3f}_{PFO:.3f}_{PFD:.3f}_{Polarity:.3f}_{ElectrodeArraySize:.3f}_{ElectrodeArrayStructure:.3f}"
def extract_parameters_from_dir_name(dirname):
    # Parse the input string
    return parse(format_witout_dir, dirname).named

def skeletonize_image(binary_img):
    # Convert to binary image using thresholding
    #_, binary_img = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    
    # Convert binary image to boolean format for skeletonization
    binary_img_bool = img_as_bool(binary_img)
    
    # Apply skeletonization
    skeleton = skeletonize(binary_img_bool)
    
    # Convert skeleton back to uint8 format for visualization
    skeleton_uint8 = (skeleton * 255).astype(np.uint8)
    
    return skeleton_uint8

def analyse_simulated_frame(image_path, SigmaX, SigmaY, width, height, total_FOV_dgrees = 15.4):
    
    # Loading the image
    image = cv2.imread(image_path)


    # Get approximate phosphene shape using the input parameteres
    get_radius_in_pixels = lambda sigma, width: round(sigma/total_FOV_dgrees*width)+1
    kernel_radius = max(get_radius_in_pixels(SigmaX, width), get_radius_in_pixels(SigmaY, height))
    kernel_size = 2 * kernel_radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    gray1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply closing
    # Apply `close` operation using a circle to approximate the phosphenes
    closed = cv2.morphologyEx(gray1, cv2.MORPH_CLOSE, kernel)
    

    # Convert to binary image using thresholding
    _, binary_img = cv2.threshold(closed, 55, 255, cv2.THRESH_BINARY)
    binary_contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

     # Create an empty image to draw and fill contours
    img_filled = np.zeros_like(binary_img)
    if not binary_contours:
        return img_filled
    # Draw and fill contours
    for contour in binary_contours:
        cv2.drawContours(img_filled, [contour], -1, (255), thickness=cv2.FILLED)
    
    # Create skeleton of the closed shape
    skeleton = skeletonize_image(img_filled)

    # Convert skeleton to uint8 format for contour finding
    skeleton_uint8 = img_as_ubyte(skeleton)
    
    #merged_contours = merge_contours(contours, distance_threshold=50)
    
    return skeleton_uint8

# Function to merge nearby contours
def merge_contours(contours, distance_threshold):
    """ Merge nearby contours within a certain distance threshold. """
    merged_contours = []
    while contours:
        contour = contours.pop(0)
        x, y, w, h = cv2.boundingRect(contour)
        # Initialize the merged contour
        merged_contour = contour
        for other_contour in contours[:]:
            x2, y2, w2, h2 = cv2.boundingRect(other_contour)
            if abs(x - x2) < distance_threshold and abs(y - y2) < distance_threshold:
                # Merge contours
                merged_contour = np.concatenate((merged_contour, other_contour), axis=0)
                contours.remove(other_contour)
        merged_contours.append(merged_contour)
    return merged_contours

def display_both_contours(img, contours1, contours2):
    # Create a new blank image with the same dimensions as the original image
    new_img = np.zeros_like(img)
     
    # Draw the first set of contours in green
    cv2.drawContours(new_img, contours1, -1, (0, 255, 0), 2)
    
    # Draw the second set of contours in red
    cv2.drawContours(new_img, contours2, -1, (0, 0, 255), 2)
    
    # Display the new image with contours
    cv2.imshow('Contours on New Image', new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def compare_contours(orig_image_path, analyzed_image_path, **kwargs):
    # Loading the image
    orig = cv2.imread(orig_image_path)

    #TODO: Params?
    orig_edges = cv2.Canny(orig, 100, 200)
    # Find contours
    contours1, _ = cv2.findContours(orig_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    analysed = analyse_simulated_frame(analyzed_image_path, **kwargs)
    contours2, _ = cv2.findContours(analysed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours2) > 1:
        print("multiple contours")
        display_both_contours(orig, contours1, contours2)

    # Ensure there is at least one contour to compare
    if contours1 and contours2:
        # print(len(contours1))
        # print(len(contours2))
        # Compare the first contour from each image
        contour1 = contours1[0]
        contour2 = contours2[0]

        # Compare contours using matchShapes
        match_score = cv2.matchShapes(contour1, contour2, cv2.CONTOURS_MATCH_I2, 0.0)
        return match_score
    else:
        return float('inf')
    
def get_generated_video(full_path):
    if not os.path.isdir(full_path):
        print(full_path)
        raise ValueError("Database error 1")
    orig_path = os.path.join(full_path, 'first_orig_frame.png')
    analysed_path = os.path.join(full_path, 'last_analysed_frame.png')
    if not os.path.isfile(orig_path):
        raise ValueError("Database error 2")
    if not os.path.isfile(analysed_path):
        raise ValueError("Database error 3")
    return orig_path, analysed_path

def iterate_generated_videos(directory_path = 'products'):
    for entry in os.listdir(directory_path):
        full_path = os.path.join(directory_path, entry)
        o, a = get_generated_video(full_path)
        yield entry, o, a
        # if os.path.isdir(full_path):
        #     orig_path = os.path.join(full_path, 'first_orig_frame.png')
        #     analysed_path = os.path.join(full_path, 'last_analysed_frame.png')
        #     if not os.path.isfile(orig_path):
        #         raise ValueError('syntesis is wrong')
        #     if not os.path.isfile(analysed_path):
        #         raise ValueError('syntesis is wrong')
        #     yield entry, orig_path,analysed_path

def reduce_suite(orig, defaults):
    for k,def_v in defaults.items():
        real_v = orig.pop(k)
        if not def_v == real_v:
            print(k)
        assert(def_v == real_v)
    return orig

default_kwargs = {
'SigmaX' : 0.13,
'SigmaY' : 0.13,
'width' : 217,
'height' : 217
}

default_values_for_suite = {'SigmaX':0.13,
    'SigmaY':0.13,
    'X0':7.0,
    'Y0':7.0,
    'DFR':60.0,
    #'EAR':10.0,
    #'PFD':10.0, # Needs to ask the author
    'Polarity':1,
    'ElectrodeArraySize':3,
    'ElectrodeArrayStructure':1
}

def extract_relevant_suite_for_original_videos(dir_name):
    params = extract_parameters_from_dir_name(dir_name)
    vid_name = params.pop('video_name')
    #print(f'suite is {params}')
    EAR = params.pop('EAR')
    PFD = params.pop('PFD')
    reduced_suite = reduce_suite(params, default_values_for_suite)
    return vid_name, reduced_suite

def get_folder_suffix_reduced_suite(reduced_suite):
    SigmaX = reduced_suite.get('SigmaX', 0.13)
    SigmaY = reduced_suite.get('SigmaY', 0.13)
    X0 = reduced_suite.get('X0', 7.0)
    Y0 = reduced_suite.get('Y0', 7.0)
    DFR = reduced_suite.get('DFR', 60.0)
    EAR = reduced_suite.get('EAR', 10.0)
    PER = reduced_suite.get('PER', 2.0)
    PFO = reduced_suite.get('PFO', 10.0)
    PFD = reduced_suite.get('PFD', 2.0)
    Polarity = reduced_suite.get('Polarity', 1)
    ElectrodeArraySize = reduced_suite.get('ElectrodeArraySize', 3)
    ElectrodeArrayStructure = reduced_suite.get('ElectrodeArrayStructure', 1)

    return f'{SigmaX:.3f}_{SigmaY:.3f}_{X0:.3f}_{Y0:.3f}_{DFR:.3f}_{EAR:.3f}_{PER:.3f}_{PFO:.3f}_{PFD:.3f}_{Polarity:.3f}_{ElectrodeArraySize:.3f}_{ElectrodeArrayStructure:.3f}'

def get_folder_name(video_name, reduced_suite):
    suite_suffix = get_folder_suffix_reduced_suite(reduced_suite)
    return f'products/{video_name}_{suite_suffix}'

def workaround_lambda(x):
    d, _1, _2 = x
    return extract_relevant_suite_for_original_videos(d)
suites_raw = list(map(workaround_lambda, iterate_generated_videos()))


# Sort the pairs by the first element
suites_raw.sort(key=itemgetter(0))
# Group by the first element and convert to a dictionary of lists
suites_dict = {key: [item[1] for item in group] for key, group in groupby(suites_raw, key=itemgetter(0))}

def get_result_files(video_name, reduced_suite):
    assert(reduced_suite in suites_dict[video_name])
    formatted_name = get_folder_name(video_name, reduced_suite)
    print(formatted_name)
    return get_generated_video(formatted_name)
    

for v in suites_dict.keys():
    for suite in suites_dict[v]:
        print(get_result_files(v, suite))
# suites={}
# lst = list(iterate_generated_videos())
# for e,_,_ in lst:
#     suites[e] = []
# for e,o,a in lst:
#     suites[e].append(extract_relevant_suite_for_original_videos(e))

#print(f'suites are {suites}')


    
    #res = compare_contours(o,a, **default_kwargs)
    #if not np.isinf(res):
    #    print(e)
    #    print(compare_contours(o,a, **default_kwargs))