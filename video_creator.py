import cv2
import numpy as np

perceptual_fading_onset = 2 # 2 seconds

def get_minimal_duration_of_video(fps, interval, num_of_directions, movement_onset):
    """ This is the length of back and forth movement starting with an onset. 
    The directions are (not normalized) vectors which are updated every frame"""
    length_in_frames = movement_onset * fps
    length_in_frames += 2*fps*interval * num_of_directions
    return length_in_frames/fps
"""
TODOs:
Friday:
1. Validate there is a single contour in all cases (that it does compare the entire shape)
4. Rewrite the python to form a pipeline (the resolution in `resize` should be a parameter)
5. Run it as pipeline on the original input videos for a benchmark
6. Run it overnight on a matrix of parameters and plot the results
Saturday:
7. Write up the results as latex.
8. Create Beamer presentation from it.
9. Create blobs with more complex features (less symmetrical) and see if it still works
Completions.
"""
def generate_random_bounded_vector(fps, interval):
    # Generate a random angle
    angle = np.random.uniform(0, 2 * np.pi)
    # See the paper, section 2.3 for the speeds measured in experiments
    max_norm = 15.4/1.5 * fps * interval
    min_norm = 15.4/10 * fps * interval
    # Generate a random magnitude within the bounds
    magnitude = np.random.uniform(min_norm, max_norm)
    
    # Compute the vector components
    x = magnitude * np.cos(angle)
    y = magnitude * np.sin(angle)
    
    return (int(x),int(y))
def generate_random_directions(fps, num, interval):
    return [generate_random_bounded_vector(fps, interval) for _ in range(num)]


def get_random_point(width, height):
    x = np.random.randint(0, width)
    y = np.random.randint(0, height)
    return (x, y)

def generate_raw_blob_points(width, height):
    num_points = np.random.randint(8, 15)  # Number of control points
    control_points = [(np.random.randint(0, width), np.random.randint(0, height)) for _ in range(num_points)]
    return np.array(control_points, dtype=np.float32)

def scale_and_center_points(points, area_ratio, width, height):
    # Calculate target area
    target_area = width * height * area_ratio
    
    # Compute the convex hull of the points
    hull = cv2.convexHull(np.array(points, dtype=np.float32))
    
    # Calculate current centroid
    moments = cv2.moments(np.array(hull, dtype=np.int32))
    if moments['m00'] == 0:
        centroid = np.array([0, 0])
    else:
        centroid = np.array([moments['m10'] / moments['m00'], moments['m01'] / moments['m00']])
    
    # Calculate target centroid (center of the image)
    target_centroid = np.array([width / 2, height / 2])
    
    # Calculate translation vector
    translation = target_centroid - centroid
    
    # Apply translation
    centered_hull = hull + translation
    
    current_area = cv2.contourArea(hull)
    if current_area == 0:
        raise ValueError("Current area of the convex hull is zero.")
    
    # Calculate scaling factor
    scale_factor = np.sqrt(target_area / current_area)
    
    # Apply scaling factor
    scaled_hull = centered_hull * scale_factor
    

    # Convert points to integer type for drawing
    scaled_hull = np.clip(scaled_hull, 0, [width-1, height-1])
    scaled_hull = np.array(scaled_hull, dtype=np.int32)

    return scaled_hull

def generate_blob_points(area_ratio, width, height):
    points = generate_raw_blob_points(width, height)    
    # Scale points to achieve the prescribed area ratio and center them
    return scale_and_center_points(points, area_ratio, width, height)

def create_moving_blob_video(output_file, width, height, fps, duration, interval, directions, movement_onset, blob_points):
    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height), isColor=False)

    num_frames = int(fps * duration)
    interval_frames = int(fps * interval)
    
    # Initialize position and directions
    x, y = width // 4, height // 4
    direction_index = 0
    dx, dy = directions[direction_index]

    frame_counter = 0
    reversed = False

    for frame_idx in range(num_frames):

        # Create a blank binary image with a black background
        frame = np.zeros((height, width), dtype=np.uint8)
        
        # Create a black image
        #frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Start movement only after onset
        if frame_idx >= int(movement_onset*fps):
            # Calculate new position
            x += dx
            y += dy

            # Reverse direction after the interval
            frame_counter += 1
            
            if frame_counter >= interval_frames:
                # Switch to next direction
                if reversed:
                    # Go to next direction
                    direction_index +=1
                    # No more directions left, stay still
                    if direction_index >= len(directions):
                        dx = 0
                        dy = 0
                    else:
                        dx, dy = directions[direction_index]
                    reversed = False
                else:
                    # Reverse the current direction to create back and forth movement
                    dx = -dx
                    dy = -dy
                    reversed = True
                # Reset the frame counter
                frame_counter = 0
            

        # Ensure the blob stays within bounds
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))

        # Apply the offset to the blob points
        moving_blob_points = np.array(blob_points, dtype=np.int32) + [x, y]
        moving_blob_points = moving_blob_points.reshape((-1, 1, 2))

        # Draw the moving blob shape
        cv2.fillPoly(frame, [moving_blob_points],255)  # White blob

        # Write the frame to the video
        video_writer.write(frame)

    # Release the video writer
    video_writer.release()
    print(f"Video saved as {output_file}")

# Parameters
image_width = 400
image_height = 400
fps = 30  # Frames per second
interval = 0.1  # Interval in seconds to change direction
directions = generate_random_directions(fps, 4, interval)
movement_onset = perceptual_fading_onset # 2 seconds of being still
area_ratio = 1/16  # Example target area ratio (e.g., 50% of image area)
duration = get_minimal_duration_of_video(fps,interval,len(directions),movement_onset)  # Duration in seconds
print(duration)
# Generate random blob points
blob_points = generate_blob_points(area_ratio, image_width, image_height)

# Creating a video or other usage can proceed with the `blob_points` as needed
# # Create the video
output_video_file = 'moving_blob_video.mp4'
create_moving_blob_video(output_video_file, image_width, image_height, fps, duration, interval, directions, movement_onset, blob_points)
