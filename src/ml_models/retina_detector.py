#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2021 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
import cv2
import time
import warnings
import numpy as np

from ibug.face_alignment import FANPredictor
from ibug.face_detection import RetinaFacePredictor

warnings.filterwarnings("ignore")


class LandmarksDetector:
    def __init__(self, device="cuda:0", model_name="mobilenet0.25"):
        self.face_detector = RetinaFacePredictor(
            device=device,
            threshold=0.8,
            model=RetinaFacePredictor.get_model(model_name),
        )
        self.landmark_detector = FANPredictor(device=device, model=None)

    def __call__(self, video_frames, skip_step=5):
        landmarks = []
        fd_time_total = 0
        ld_time_total = 0
        h, w, _ = video_frames[0].shape
        h_new, w_new = h, w
        while min(h_new, w_new) >= 512: # change to if to resize it only 2 times, otherwise it will be the power of 2
            h_new = h // 2
            w_new = w // 2
        
        for i, frame in enumerate(video_frames):
            if i != len(video_frames) - 1 and i % skip_step != 0:
                landmarks.append(None)
                continue
            cur_time = time.time()
            resized_frame = cv2.resize(frame, (w_new, h_new))     
            detected_faces = self.face_detector(resized_frame, rgb=False)
            fd_time_total += time.time() - cur_time
            cur_time = time.time()
            face_points, _ = self.landmark_detector(resized_frame, detected_faces, rgb=True)
            if len(detected_faces) == 0:
                landmarks.append(None)
            else:
                max_id, max_size = 0, 0
                for idx, bbox in enumerate(detected_faces):
                    bbox_size = (bbox[2] - bbox[0]) + (bbox[3] - bbox[1])
                    if bbox_size > max_size:
                        max_id, max_size = idx, bbox_size
                max_face_points: np.ndarray = face_points[max_id]
                rescaled_face_points = max_face_points.copy()
                rescaled_face_points[:, 0] = rescaled_face_points[:, 0] * (h // h_new)
                rescaled_face_points[:, 1] = rescaled_face_points[:, 1] * (w // w_new)
                landmarks.append(rescaled_face_points)
            ld_time_total += time.time() - cur_time
        print('FD:', fd_time_total)
        print('LD:', ld_time_total)

        return landmarks
