import cv2
import os
from contextlib import contextmanager

@contextmanager
def video_capture_context(video_path):
    # Create a VideoCapture object
    cap = cv2.VideoCapture(video_path)
    try:
        # Ensure the video file was opened successfully
        if not cap.isOpened():
            raise RuntimeError("Error: Could not open video.")
        yield cap
    finally:
        # Release the VideoCapture object
        cap.release()


#To be used where `cap` being a context manager of `cv2.VideoCapture`
def extract_frame_at_time(cap, inserted_time = 0, input_in_seconds = False):

    fps = cap.get(cv2.CAP_PROP_FPS)
    if input_in_seconds:
        inserted_time *= fps
    # Get the total number of frames
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if inserted_time < 0:
        frame_number = frame_count + inserted_time
    else:
        # Set the frame number you want to extract
        frame_number = int(inserted_time * fps)  # Change this to the frame number you want

    # Set the frame position in the video
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Read the frame
    ret, frame = cap.read()

    # Check if the frame was read successfully
    if not ret:
        raise ValueError("Could not read frame")

    # Optionally, save the frame to a file
    return frame_number, frame


# video_name = 'with_static_start'
# perceptual_fading_duration = 2.0 #seconds
# perceptual_fading_onset=0.5 # seconds
# persistence_duration = 3.0 #seconds

# # Open the video file
# video_path = f'{video_name}.mp4'
# with video_capture_context(video_path) as cap:

#     # Create a directory to save frames
#     output_dir = f'frames_of_{video_name}'
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     #First frame when object begins to fade
#     frame_id, frame = extract_frame_at_time(cap, perceptual_fading_duration)

#     frame_filename = os.path.join(output_dir, f'{frame_id:04d}.png')
#     cv2.imwrite(frame_filename, frame)

# print(f"Frame extracted and saved to {output_dir}.")

