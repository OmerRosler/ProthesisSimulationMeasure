import cv2
import os

def create_video_from_images(image_list, output_video_file, fps):
    # Check if the list is empty
    if not image_list:
        raise ValueError("The image list is empty.")

    # Read the first image to get the size
    first_image = cv2.imread(image_list[0])
    height, width, _ = first_image.shape

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    video_writer = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height))

    for image_file in image_list:
        # Read the image
        img = cv2.imread(image_file)
        if img is None:
            print(f"Warning: Image {image_file} could not be read.")
            continue
        
        # Check if the image size matches the first image
        if (img.shape[1], img.shape[0]) != (width, height):
            raise ValueError(f"Image size does not match the first image: {image_file}")

        # Write the image to the video
        video_writer.write(img)

    # Release the VideoWriter
    video_writer.release()
    print(f"Video saved as {output_video_file}")

def file_iterator(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            yield file_path
# Example usage
dir_name = 'out_2_0.5_fading'
image_files = list(file_iterator(dir_name)) # List of image file paths
output_file = f'output_dir_{dir_name}.mp4'  # Output video file path
fps = 60  # Frames per second

create_video_from_images(image_files, output_file, fps)
