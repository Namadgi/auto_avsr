import sys
import torchvision
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
# t = torchvision.io.read_video('data/test_movie.MOV', pts_unit="sec")[0]
# print(t.shape)
vidcap = cv2.VideoCapture('data/test_movie.MOV')
images = []
success, image = vidcap.read()
count = 0
while success:
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    images.append(image)
    #   cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
    success, image = vidcap.read()
    # print('Read a new frame: ', success)
    count += 1

images_np = torch.from_numpy(np.array(images)[:, :, :, ::-1].copy())
print(images_np.shape)
plt.imshow(images_np[0].numpy())
plt.show()