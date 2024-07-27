import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def reduce_resolution(image_path):
    """ Convert image to "posphene based" where there was 
    """
    image = cv2.imread(image_path)
    # Loading the image

    gray1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    orig_blurreed = cv2.GaussianBlur(gray1, (5, 5), 0)

    half = cv2.resize(gray1, (16, 16))
    #gray = cv2.cvtColor(half, cv2.COLOR_BGR2GRAY)
    (thresh, im_bw) = cv2.threshold(half, 10, 255, cv2.THRESH_BINARY)
    # Step 2: Check the number of channels

        
    # Resize image to a larger size
    scale_factor = 10  # Example scale factor
    image_resized = cv2.resize(im_bw, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)

    # Apply Gaussian blur
    image_blurred = cv2.GaussianBlur(image_resized, (5, 5), 0)

    # Titles =["Original", "Half", "Threshould", "Rescaled", "Only blur"]
    # images =[image, half, im_bw, image_blurred, orig_blurreed]
    # count = len(Titles)

    # for i in range(count):
    #     plt.subplot(3, 3, i + 1)
    #     plt.title(Titles[i])
    #     plt.imshow(images[i], cmap='gray')

    # plt.show()
    return image_blurred

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

def compare_contours(orig_image_path, analyzed_image_path):

    orig_reduced = reduce_resolution(orig_image_path)
    analyzed_reduced = reduce_resolution(analyzed_image_path)
    # Apply thresholding or edge detection
    # Using Canny edge detection
    orig_edges = cv2.Canny(orig_reduced, 100, 200)
    # Find contours
    contours1, _ = cv2.findContours(orig_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Apply thresholding or edge detection
    # Using Canny edge detection
    analyzed_edges = cv2.Canny(analyzed_reduced, 100, 200)
    # Find contours
    contours2, _ = cv2.findContours(analyzed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #TODO: Is this needed?
    # Merge nearby contours if needed
    #merged_contours1 = merge_contours(contours1, distance_threshold=50)  # Adjust the distance threshold as needed
    #merged_contours2 = merge_contours(contours2, distance_threshold=50)  # Adjust the distance threshold as needed

    # Draw contours on the original images for visualization
    cv2.drawContours(orig_reduced, contours1, -1, (0, 255, 0), 3)
    cv2.drawContours(analyzed_reduced, contours2, -1, (0, 255, 0), 3)


    # Display the images with contours
    # cv2.imshow('Image 1 Contours', orig_reduced)
    # cv2.imshow('Image 2 Contours', analyzed_reduced)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
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

# product_dir_with_per_3 = "products/HeadMovementEllipse_0.130_0.130_7.000_7.000_60.000_10.000_3.000_1.000_2.000_1.000_3.000_1.000"
# product_dir_with_per_0 = "products/HeadMovementEllipse_0.130_0.130_7.000_7.000_60.000_10.000_3.000_1.000_2.000_1.000_3.000_1.000"
# orig_image = os.path.join(product_dir, 'first_orig_frame.png')
# analysed_image = os.path.join(product_dir, 'last_analysed_frame.png')
# print(compare_contours(orig_image, analysed_image))