import random
from src.helpers import get_frames_from_video_file, sample_frames_from_video_file
import numpy as np


class FrameGenerator:
    def __init__(self, path, frame_count, output_shape=(256, 256), training=False):
        """ Returns a set of frames with their associated label.

          Args:
            path: Video file paths.
            n_frames: Number of frames.
            training: Boolean to determine if training dataset is being created.
        """
        self.path = path
        self.frame_count = frame_count
        self.training = training
        self.output_shape = output_shape
        self.class_names = sorted(set(p.name for p in self.path.iterdir() if p.is_dir()))
        self.class_ids_for_name = dict((name, idx) for idx, name in enumerate(self.class_names))
        print("Detected Labels: ", self.class_ids_for_name)

    def get_files_and_class_names(self):
        video_paths = list(self.path.glob('*/*.mp4'))
        classes = [p.parent.name for p in video_paths]
        return video_paths, classes

    def __call__(self):
        video_paths, classes = self.get_files_and_class_names()

        pairs = list(zip(video_paths, classes))

        if self.training:
            random.shuffle(pairs)

        for path, name in pairs:
            video_frames = get_frames_from_video_file(path, self.frame_count, output_size=self.output_shape)
            label = [self.class_ids_for_name[name]]  # Encode labels
            yield video_frames, label


class SamplingFrameGenerator:
    def __init__(self, path, sample_count, frames_per_sample, output_shape=(256, 256), training=False):
        """ Returns a set of frames with their associated label.

          Args:
            path: Video file paths.
            n_frames: Number of frames.
            training: Boolean to determine if training dataset is being created.
        """
        self.path = path
        self.sample_count = sample_count
        self.frames_per_sample = frames_per_sample
        self.training = training
        self.output_shape = output_shape
        self.class_names = sorted(set(p.name for p in self.path.iterdir() if p.is_dir()))
        self.class_ids_for_name = dict((name, idx) for idx, name in enumerate(self.class_names))
        print("Detected Labels: ", self.class_ids_for_name)

    def get_files_and_class_names(self):
        video_paths = list(self.path.glob('*/*.mp4'))
        classes = [p.parent.name for p in video_paths]
        return video_paths, classes

    def __call__(self):
        video_paths, classes = self.get_files_and_class_names()

        pairs = list(zip(video_paths, classes))

        if self.training:
            random.shuffle(pairs)

        for path, name in pairs:
            video_frames = sample_frames_from_video_file(path, self.sample_count, self.frames_per_sample, output_size=self.output_shape)
            label = [self.class_ids_for_name[name]]  # Encode labels
            yield video_frames, label
