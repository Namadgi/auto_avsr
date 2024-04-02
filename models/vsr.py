import os
import torch
import torchvision
import cv2
import numpy as np
import time
from data_transforms import VideoTransform
from retina_detector import LandmarksDetector
from retina_video_process import VideoProcess

from model_module import ModelModule

class VSR(torch.nn.Module):
    def __init__(self):
        super(VSR, self).__init__()
        
        self.landmarks_detector = LandmarksDetector(device="cuda")
        self.video_process = VideoProcess(convert_gray=False)
        self.video_transform = VideoTransform(subset="test")
        self.modelmodule = ModelModule()


    def forward(self, data_filename):
        data_filename = os.path.abspath(data_filename)
        cur_time = time.time()
        video = self.load_video(data_filename)
        print('IP: ', time.time() - cur_time)
        cur_time = time.time()
        landmarks = self.landmarks_detector(video)
        print('FD: ', time.time() - cur_time)
        cur_time = time.time()
        video = self.video_process(video, landmarks)
        print('FC: ', time.time() - cur_time)
        if video is None:
            return ''
        # print(video, type(video))
        video = torch.tensor(video)
        video = video.permute((0, 3, 1, 2))
        cur_time = time.time()
        video = self.video_transform(video)
        print('VE: ', time.time() - cur_time)
        cur_time = time.time()
        with torch.no_grad():
            transcript = self.modelmodule(video)
        print('VI: ', time.time() - cur_time)
        return transcript

    def load_video(self, data_filename):
        # cap = cv2.VideoCapture(data_filename)
        # length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # images = []
        # success, image = cap.read()
        # # count = 0
        # while success:
        #     # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #     images.append(image)
        #     #   cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
        #     success, image = cap.read()
        #     # print('Read a new frame: ', success)
        #     # count += 1
        # images_np = torch.from_numpy(np.array(images)[:, :, :, ::-1].copy())
        # return images_np


        return torchvision.io.read_video(data_filename, end_pts=10, pts_unit="sec")[0].numpy()


# if __name__ == "__main__":
#     pipeline = VSR()
#     pipeline.load_state_dict(torch.load('../model_weights/vsr.pth'))
#     pipeline = pipeline.to('cuda')
#     pipeline.eval()
#     transcript = pipeline('../data/one_ten.MOV')
#     print(f"transcript: {transcript}")