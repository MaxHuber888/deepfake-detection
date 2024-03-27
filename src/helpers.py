import os
import cv2
import pandas as pd
from pytube import YouTube
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt


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


def videos_to_frames(vid_path, frame_path, frame_gap, max_frames=-1):
    video_paths = glob(f"{vid_path}/*")

    for path in tqdm(video_paths):
        name = path.split("/")[-1].split(".")[0]
        print(name)

        save_path = os.path.join(frame_path, name)

        try:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
        except OSError:
            print(f"ERROR: creating directory with name {save_path}")

        cap = cv2.VideoCapture(path)
        idx = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                cap.release()
                break

            if idx == 0:
                cv2.imwrite(f"{save_path}/{idx}.png", frame)
            else:
                if idx % frame_gap == 0:
                    cv2.imwrite(f"{save_path}/{idx}.png", frame)
                if idx == max_frames and max_frames > 0:
                    cap.release()
                    break

            idx += 1


def load_frames(frames_path, batch_size=1):
    frame_array = []

    frame_paths = glob(f"{frames_path}/*")

    for frame_dir in tqdm(frame_paths):
        i = 0
        frames = glob(f"{frame_dir}/*.png")
        frame_list = []
        for frame in frames:
            i += 1
            image = cv2.imread(frame)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imshow('image', rgb_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            frame_list.append(rgb_image)

            if i == batch_size:
                break

        frame_array.append(frame_list)
        break

    return frame_array


def show_image(image):
    plt.imshow(image)
    plt.show()
