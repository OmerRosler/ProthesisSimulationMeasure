import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage import img_as_bool, img_as_ubyte


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
    
    
def iterate_generated_videos(directory_path = 'products'):
    for entry in os.listdir(directory_path):
        full_path = os.path.join(directory_path, entry)
        if os.path.isdir(full_path):
            orig_path = os.path.join(full_path, 'first_orig_frame.png')
            analysed_path = os.path.join(full_path, 'last_analysed_frame.png')
            if not os.path.isfile(orig_path):
                raise ValueError('syntesis is wrong')
            if not os.path.isfile(analysed_path):
                raise ValueError('syntesis is wrong')
            yield entry, orig_path,analysed_path

default_kwargs = {
'SigmaX' : 0.13,
'SigmaY' : 0.13,
'width' : 217,
'height' : 217
}
for e,o,a in iterate_generated_videos():
    res = compare_contours(o,a, **default_kwargs)
    if not np.isinf(res):
        print(e)
        print(compare_contours(o,a, **default_kwargs))