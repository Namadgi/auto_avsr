import os
import torch
import torchvision

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


if __name__ == "__main__":
    pipeline = VSR()
    pipeline.load_state_dict(torch.load('../model_weights/vsr_pipe.pth'))
    pipeline = pipeline.to('cuda')
    pipeline.eval()
    transcript = pipeline('../data/one_ten.MOV')
    print(f"transcript: {transcript}")