import os
import cv2
from itertools import groupby
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial.distance import cdist

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

def analyse_simulated_frame(image_path, SigmaX, SigmaY, width, height, total_FOV_dgrees = 15.4, generate_image_compariosn = False):
    
    # Loading the image
    image = cv2.imread(image_path)

    if generate_image_compariosn:
        subplot_images = []
        subplot_names = ['Original', 'Closed', 'Skeletonized', 'Merged']
        subplot_images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

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

    for contour in binary_contours:
        cv2.drawContours(img_filled, [contour], -1, (255), thickness=cv2.FILLED)
    #cv2.drawContours(img_filled, [final_contour], -1, (255), thickness=cv2.FILLED)
    
    if generate_image_compariosn:
        subplot_images.append(cv2.cvtColor(img_filled, cv2.COLOR_BGR2RGB))

    # Create skeleton of the closed shape
    skeleton = skeletonize_image(img_filled)

    # Convert skeleton to uint8 format for contour finding
    skeleton_uint8 = img_as_ubyte(skeleton)

    if generate_image_compariosn:
        subplot_images.append(cv2.cvtColor(skeleton_uint8, cv2.COLOR_BGR2RGB))
    
    contours2, _ = cv2.findContours(skeleton_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    merged2 = merge_contours(contours2)


    if generate_image_compariosn:
        empty_image = np.zeros_like(subplot_images[0])
        # Draw the convex hull
        cv2.polylines(empty_image, [merged2], isClosed=True, color=(0, 255, 0), thickness=2)

        subplot_images.append(cv2.cvtColor(empty_image, cv2.COLOR_BGR2RGB))
        
        
        #subplot_images.append(cv2.cvtColor(merged2, cv2.COLOR_BGR2RGB))
        
        # Create a grid of subplots (2 rows and 2 columns)
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))

        # Flatten the axs array for easy iteration
        axs = axs.flatten()

        # Plot each image in a subplot with titles
        for ax, img, title in zip(axs, subplot_images, subplot_names):
            ax.imshow(img)  # Convert BGR to RGB
            ax.set_title(title)  # Set title for each subplot
            ax.axis('off')  # Hide the axes

        image_dir = os.path.dirname(image_path)
        fig_file_name = os.path.join(image_dir, 'compare_pipeline')
        fig.savefig(fig_file_name)
        plt.show()
        plt.close(fig)
    return merged2

def merge_contours(contours):
    list_of_pts = [] 
    for ctr in contours:
        list_of_pts += [pt[0] for pt in ctr]
    origin = np.array(list_of_pts).mean(axis = 0) # get origin
    clock_ang_dist = clockwise_angle_and_distance(origin) # set origin
    list_of_pts = sorted(list_of_pts, key=clock_ang_dist) # use to sort
    ctr = np.array(list_of_pts).reshape((-1,1,2)).astype(np.int32)
    ctr = cv2.convexHull(ctr)
    return ctr

class clockwise_angle_and_distance():
    '''
    A class to tell if point is clockwise from origin or not.
    This helps if one wants to use sorted() on a list of points.

    Parameters
    ----------
    point : ndarray or list, like [x, y]. The point "to where" we g0
    self.origin : ndarray or list, like [x, y]. The center around which we go
    refvec : ndarray or list, like [x, y]. The direction of reference

    use: 
        instantiate with an origin, then call the instance during sort
    reference: 
    https://stackoverflow.com/questions/41855695/sorting-list-of-two-dimensional-coordinates-by-clockwise-angle-using-python

    Returns
    -------
    angle
    
    distance
    

    '''
    def __init__(self, origin):
        self.origin = origin

    def __call__(self, point, refvec = [0, 1]):
        if self.origin is None:
            raise NameError("clockwise sorting needs an origin. Please set origin.")
        # Vector between point and the origin: v = p - o
        vector = [point[0]-self.origin[0], point[1]-self.origin[1]]
        # Length of vector: ||v||
        lenvector = np.linalg.norm(vector[0] - vector[1])
        # If length is zero there is no angle
        if lenvector == 0:
            return -np.pi, 0
        # Normalize vector: v/||v||
        normalized = [vector[0]/lenvector, vector[1]/lenvector]
        dotprod  = normalized[0]*refvec[0] + normalized[1]*refvec[1] # x1*x2 + y1*y2
        diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1] # x1*y2 - y1*x2
        angle = np.arctan2(diffprod, dotprod)
        # Negative angles represent counter-clockwise angles so we need to 
        # subtract them from 2*pi (360 degrees)
        if angle < 0:
            return 2*np.pi+angle, lenvector
        # I return first the angle because that's the primary sorting criterium
        # but if two vectors have the same angle then the shorter distance 
        # should come first.
        return angle, lenvector


def display_both_contours(img, contours1, contours2):
    # Create a new blank image with the same dimensions as the original image
    new_img = np.zeros_like(img)
     
    # Draw the first set of contours in green
    cv2.drawContours(new_img, contours1, -1, (0, 255, 0), 2)
    
    # Draw the second set of contours in red
    cv2.polylines(new_img, [contours2], isClosed=True, color=(0, 0, 255), thickness=2)
    #cv2.drawContours(new_img, contours2, -1, (0, 0, 255), 2)
    
    # Display the new image with contours
    cv2.imshow('Contours on New Image', new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def align_and_normalize_contours(cnt1, cnt2):
    # Compute the centroids
    M1 = cv2.moments(cnt1)
    M2 = cv2.moments(cnt2)
    centroid1 = np.array([M1['m10'] / M1['m00'], M1['m01'] / M1['m00']])
    centroid2 = np.array([M2['m10'] / M2['m00'], M2['m01'] / M2['m00']])
    
    # Align contours to their centroids
    aligned_cnt1 = cnt1 - centroid1
    aligned_cnt2 = cnt2 - centroid2
    
    # Normalize the contours to the same scale
    norm_factor1 = np.linalg.norm(aligned_cnt1)
    norm_factor2 = np.linalg.norm(aligned_cnt2)
    normalized_cnt1 = aligned_cnt1 / norm_factor1
    normalized_cnt2 = aligned_cnt2 / norm_factor2
    
    return normalized_cnt1, normalized_cnt2

def measure_of_difference(contour1, contour2):
    return cv2.matchShapes(contour1, contour2, cv2.CONTOURS_MATCH_I2, 0.0)

def measure_of_difference2(contour1, contour2):
    # Align and normalize the contours
    aligned_cnt1, aligned_cnt2 = align_and_normalize_contours(contour1.squeeze(), contour2.squeeze())

    # Ensure both contours have the same number of points by resampling if necessary
    num_points = min(len(aligned_cnt1), len(aligned_cnt2))
    aligned_cnt1 = cv2.resize(aligned_cnt1, (num_points, 2), interpolation=cv2.INTER_LINEAR)
    aligned_cnt2 = cv2.resize(aligned_cnt2, (num_points, 2), interpolation=cv2.INTER_LINEAR)

    # Compute the Euclidean distance between corresponding points
    distances = cdist(aligned_cnt1, aligned_cnt2, 'euclidean')
    mean_distance = np.mean(np.diagonal(distances))
    return mean_distance
def compare_contours(orig_image_path, analyzed_image_path, **kwargs):
    # Loading the image
    orig = cv2.imread(orig_image_path)

    #TODO: Params?
    orig_edges = cv2.Canny(orig, 100, 200)
    # Find contours
    contours1, _ = cv2.findContours(orig_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    analysed = analyse_simulated_frame(analyzed_image_path, **kwargs)
    
    # Ensure there is at least one contour to compare
    if not contours1:
        return float('inf')
    contour1 = contours1[0]
    #display_both_contours(orig, contours1, merged2)
    # Compare contours using matchShapes
    display_both_contours(orig,contour1, analysed)
    match_score = measure_of_difference2(contour1, analysed)
    return match_score

    
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

# def reduce_suite(orig, defaults):
#     for k,def_v in defaults.items():
#         real_v = orig.pop(k)
#         if not def_v == real_v:
#             print(k)
#         assert(def_v == real_v)
#     return orig

# These are always used for the dataset we ran on
default_kwargs = {
'SigmaX' : 0.13,
'SigmaY' : 0.13,
'width' : 217,
'height' : 217
}

# default_values_for_suite = {'SigmaX':0.13,
#     'SigmaY':0.13,
#     'X0':7.0,
#     'Y0':7.0,
#     'DFR':60.0,
#     #'EAR':10.0,
#     #'PFD':10.0, # Needs to ask the author
#     'Polarity':1,
#     'ElectrodeArraySize':3,
#     'ElectrodeArrayStructure':1
# }

# def extract_relevant_suite_for_original_videos(dir_name):
#     params = extract_parameters_from_dir_name(dir_name)
#     vid_name = params.pop('video_name')
#     #print(f'suite is {params}')
#     EAR = params.pop('EAR')
#     PFD = params.pop('PFD')
#     reduced_suite = reduce_suite(params, default_values_for_suite)
#     return vid_name, reduced_suite

# def get_folder_suffix_reduced_suite(reduced_suite):
#     SigmaX = reduced_suite.get('SigmaX', 0.13)
#     SigmaY = reduced_suite.get('SigmaY', 0.13)
#     X0 = reduced_suite.get('X0', 7.0)
#     Y0 = reduced_suite.get('Y0', 7.0)
#     DFR = reduced_suite.get('DFR', 60.0)
#     EAR = reduced_suite.get('EAR', 10.0)
#     PER = reduced_suite.get('PER', 2.0)
#     PFO = reduced_suite.get('PFO', 10.0)
#     PFD = reduced_suite.get('PFD', 2.0)
#     Polarity = reduced_suite.get('Polarity', 1)
#     ElectrodeArraySize = reduced_suite.get('ElectrodeArraySize', 3)
#     ElectrodeArrayStructure = reduced_suite.get('ElectrodeArrayStructure', 1)

#     return f'{SigmaX:.3f}_{SigmaY:.3f}_{X0:.3f}_{Y0:.3f}_{DFR:.3f}_{EAR:.3f}_{PER:.3f}_{PFO:.3f}_{PFD:.3f}_{Polarity:.3f}_{ElectrodeArraySize:.3f}_{ElectrodeArrayStructure:.3f}'

# def get_folder_name(video_name, reduced_suite):
#     suite_suffix = get_folder_suffix_reduced_suite(reduced_suite)
#     return f'products/{video_name}_{suite_suffix}'

# def workaround_lambda(x):
#     d, _1, _2 = x
#     return extract_relevant_suite_for_original_videos(d)
# suites_raw = list(map(workaround_lambda, iterate_generated_videos()))


# # Sort the pairs by the first element
# suites_raw.sort(key=itemgetter(0))
# # Group by the first element and convert to a dictionary of lists
# suites_dict = {key: [item[1] for item in group] for key, group in groupby(suites_raw, key=itemgetter(0))}

# def get_result_files(video_name, reduced_suite):
#     assert(reduced_suite in suites_dict[video_name])
#     formatted_name = get_folder_name(video_name, reduced_suite)
#     return get_generated_video(formatted_name)
    

# for v in suites_dict.keys():
#     for suite in suites_dict[v]:
#         print(v, suite)
#         orig, anal = get_result_files(v, suite)
#         print(compare_contours(orig,anal, **default_kwargs))
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
def compare(outfile = 'run2_raw_results.txt'):
    with open(outfile, 'w') as file:
        for e,o,a in iterate_generated_videos('run2'):
            res = compare_contours(o, a, **default_kwargs)
            file.write(f'{e}, {res}\n')

#compare()

def compare3(outfile = 'run3_raw_results.txt'):
    #with open(outfile, 'w') as file:
    for e,o,a in iterate_generated_videos('run3'):
        res = compare_contours(o, a, **default_kwargs, generate_image_compariosn = False)
        print(res)
        #file.write(f'{e}, {res}\n')

compare3()


# directory = 'products'
# for d,o,a in iterate_generated_videos(directory):
#     compare_contours(o,a, generate_image_compariosn = True, **default_kwargs)
# for entry in os.listdir(directory):
#     full_path = os.path.join(directory, entry)
#     # Check if the entry is a directory
#     if os.path.isdir(full_path):
#         print(full_path)

# def remove_noise_from_name(name):
#     format_start = 'run2_size_{start}_'

# def find_all_absolute_no_detection(infile= 'run2_raw_results.txt'):
#     with open(infile, 'r') as file:
#         for line in file:
#             name, value = line.strip().split(', ')
#             if value == "1.7976931348623157e+308":
#                 print(name)

# find_all_absolute_no_detection()