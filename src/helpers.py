import os
import random
import cv2
import imageio
import pandas as pd
from tensorflow_docs.vis import embed
from pytube import YouTube
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


def url_to_videos(url_list, vid_path):
    for link in tqdm(url_list):
        try:
            # Create link
            yt = YouTube(link)
            # Get all streams and filter for mp4 files
            mp4_streams = yt.streams.filter(file_extension='mp4', progressive=True)
            # Get the video with the highest resolution
            d_video = mp4_streams[-1]
            # Download the video
            d_video.download(output_path=vid_path)
        except Exception as e:
            print(e)

    print('Task Completed!')


def show_image(image):
    plt.imshow(image)
    plt.show()


def format_frames(frame, output_size):
    frame = tf.image.convert_image_dtype(frame, tf.float32)
    frame = tf.image.resize_with_pad(frame, *output_size)
    return frame


# Returns numpy array of frames
def get_frames_from_video_file(video_path, frame_count, output_size=(256, 256), frame_step=10):
    # Read each video frame by frame
    result = []
    src = cv2.VideoCapture(str(video_path))

    video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)

    need_length = 1 + (frame_count - 1) * frame_step

    if need_length > video_length:
        start = 0
    else:
        max_start = video_length - need_length
        start = random.randint(0, max_start + 1)

    src.set(cv2.CAP_PROP_POS_FRAMES, start)
    # ret is a boolean indicating whether read was successful, frame is the image itself
    ret, frame = src.read()
    result.append(format_frames(frame, output_size))

    for _ in range(frame_count - 1):
        for _ in range(frame_step):
            ret, frame = src.read()
        if ret:
            frame = format_frames(frame, output_size)
            result.append(frame)
        else:
            result.append(np.zeros_like(result[0]))
    src.release()
    result = np.array(result)[..., [2, 1, 0]]

    return result


def sample_frames_from_video_file(video_path, sample_count, frames_per_sample, frame_step=10, output_size=(256, 256)):
    # Read each video frame by frame
    result = []
    src = cv2.VideoCapture(str(video_path))

    video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)

    need_length = 1 + (frames_per_sample - 1) * frame_step

    max_start = video_length - need_length

    sample_starts = []

    for sample in range(sample_count):
        sample_start = int(max_start * sample / sample_count)
        sample_starts.append(sample_start)
        #print(sample_start)

    for start in sample_starts:
        src.set(cv2.CAP_PROP_POS_FRAMES, start)
        # ret is a boolean indicating whether read was successful, frame is the image itself
        ret, frame = src.read()
        result.append(format_frames(frame, output_size))

        for _ in range(frames_per_sample - 1):
            for _ in range(frame_step):
                ret, frame = src.read()
            if ret:
                frame = format_frames(frame, output_size)
                result.append(frame)
            else:
                result.append(np.zeros_like(result[0]))
    src.release()
    result = np.array(result)[..., [2, 1, 0]]

    return result


def to_gif(images):
    converted_images = np.clip(images * 255, 0, 255).astype(np.uint8)
    imageio.mimsave('./animation.gif', converted_images, fps=10)
    return embed.embed_file('./animation.gif')


def train_val_split(source_path, split_path, train_test_ratio=0.8):
    class_list = os.listdir(source_path)

    if len(class_list) > 0:
        for label in class_list:
            # Get list of videos in current label/class
            current_class_vids = os.listdir(source_path + "/" + label)
            print("Label:", label)
            print(current_class_vids[:3], "\n")

            # Randomly split into train/test
            current_class_df = pd.DataFrame(current_class_vids)

            train_df = current_class_df.sample(frac=train_test_ratio)
            train_list = train_df.values.tolist()
            test_list = [item for item in current_class_vids if item not in train_list]

            # Copy files to the correct destination
            # TODO

    print("Done.")

def save_history(history, model_name):
    hist_df = pd.DataFrame(history.history)
    # If previous history exists, concatenate histories
    if os.path.exists("saved_models/" + model_name + "_history.csv"):
        previous_hist_df = pd.read_csv("saved_models/" + model_name + "_history.csv")
        hist_df = pd.concat([previous_hist_df, hist_df], axis=0, ignore_index=True)
    hist_df.to_csv("saved_models/" + model_name + "_history.csv")
