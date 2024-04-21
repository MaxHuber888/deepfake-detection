import os

import cv2
import numpy as np
from tqdm import tqdm


def compute_optical_flow(video_path, output_path, show_video=False):
    # Capture the video from the input file
    cap = cv2.VideoCapture(video_path)

    # Determine the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get video frame dimensions and fps to use in the output
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create a VideoWriter object to write the video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        output_path, fourcc, fps, (frame_width, frame_height), isColor=True
    )

    # read the first frame
    ret, prev_frame = cap.read()
    if not ret:
        print(f"Can't receive frame (stream end?) from {video_path}. Exiting ...")
        cap.release()
        out.release()
        return

    # convert frame to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # progress bar
    pbar = tqdm(
        total=total_frames,
        desc=f"Processing {os.path.basename(video_path)}",
        unit="frame",
    )

    # Process video
    while True:
        ret, frame = cap.read()
        if not ret:
            pbar.close()
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        # Compute magnitude and angle of 2D vectors
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Normalize the magnitude to range 0 to 1
        magnitude = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)

        # Create an RGB image for the output to match tensor input
        mask = np.zeros_like(prev_frame)
        mask[..., 1] = 255
        mask[..., 0] = angle * 180 / np.pi / 2
        mask[..., 2] = magnitude * 255

        # Convert HSV to RGB
        rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

        # Write the output frame
        out.write(rgb)

        # Display the frame
        if show_video:
            cv2.imshow("frame", rgb)
            if cv2.waitKey(1) == ord("q"):
                break

        # Update previous frame and previous gray
        prev_gray = gray

        # Update the progress bar
        pbar.update(1)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def process_directory(directory_path):
    for filename in tqdm(
        os.listdir(directory_path), desc="Getting Optical Flow For Files"
    ):
        if filename.lower().endswith((".mp4")):  # check for video file extensions
            video_path = os.path.join(directory_path, filename)
            output_path = os.path.join(directory_path, f"output_{filename}")
            compute_optical_flow(video_path, output_path, show_video=False)


# NOTE REPLACE directory_path with the path to whatever video directory you want
directory_path = "path_to_video_directory"
process_directory(directory_path)
