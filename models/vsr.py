import os
import torch
import torchvision
from datamodule.transforms import VideoTransform
from preparation.detectors.retinaface.detector import LandmarksDetector
from preparation.detectors.retinaface.video_process import VideoProcess
from lightning import ModelModule

class InferencePipeline(torch.nn.Module):
    def __init__(self):
        super(InferencePipeline, self).__init__()
        
        self.landmarks_detector = LandmarksDetector(device="cuda")
        self.video_process = VideoProcess(convert_gray=False)
        self.video_transform = VideoTransform(subset="test")

        self.modelmodule = ModelModule(cfg)


    def forward(self, data_filename):
        data_filename = os.path.abspath(data_filename)

        video = self.load_video(data_filename)
        landmarks = self.landmarks_detector(video)
        video = self.video_process(video, landmarks)
        video = torch.tensor(video)
        video = video.permute((0, 3, 1, 2))
        video = self.video_transform(video)

        with torch.no_grad():
            transcript = self.modelmodule(video)

        return transcript

    def load_video(self, data_filename):
        return torchvision.io.read_video(data_filename, pts_unit="sec")[0].numpy()
