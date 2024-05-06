import os
import numpy as np
import json

import torch
from ts.torch_handler.base_handler import BaseHandler

import subprocess
import time
import strsimpy

SCALE = 0.25
VIDEO_TEMP   = 'temp.avi'
VIDEO_OUTPUT = 'output.mp4'
VIDEO_FACE_D = 'output.avi'
# AUDIO_OUTPUT = 'output.wav'

class VSRHandler(BaseHandler):
    NLEV_THRESHOLD = 0.5
    
    def preprocess(self, data):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        
        if data is None:
            return data

        if os.path.isfile(VIDEO_TEMP):
            os.remove(VIDEO_TEMP)
        
        if os.path.isfile(VIDEO_OUTPUT):
            os.remove(VIDEO_OUTPUT)
        
        for row in data:
            data = row.get('data') or row.get('body')

        is_dict = True
        if isinstance(data, dict):
            data = data['instances'][0]
        else:
            try:
                data = json.loads(data)
            except:
                is_dict = False
                with open(VIDEO_TEMP, 'wb') as out_file:
                    out_file.write(data)
                    object_name = VIDEO_TEMP
        phrase = ''
        if is_dict:
            # Download file
            token = data['token']
            bucket_name = data['bucket_name']
            object_name = data['object_name']
            phrase = data['phrase']
            os.system(
                f'curl -X GET ' +
                f'-H "Authorization: Bearer {token}" -o {object_name} '
                f'"https://storage.googleapis.com/storage/v1/b/{bucket_name}/o/{object_name}?alt=media"'
            )

        subprocess.call(
            f'ffmpeg -y -i {object_name} -qscale:v 2 -threads 10 ' +\
            f'-async 1 -r 25 -vf scale="-2:640" {VIDEO_OUTPUT} -loglevel panic',
            # f'ffmpeg -y -i {object_name} -r 24 {VIDEO_OUTPUT}',
            shell=True, 
            stdout=None,    
        )

        # subprocess.call(
        #     f'ffmpeg -y -i {VIDEO_OUTPUT} -vf scale="-2:640" {VIDEO_FACE_D}',
        #     shell=True,
        #     stdout=None
        # )

        return VIDEO_OUTPUT, phrase

    def inference(self, file):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        # Do some inference call to engine here and return output
        self.model.eval()
        with torch.no_grad():
            y = self.model(file)
        return y

    def postprocess(self, pred_text, target_text):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """

        descriptions = [
            'Successful check',
            'Phrase does not match',
            'No face present',
        ]

        if pred_text == '':
            code = 2
        else:
            code = int(self.compare_texts(pred_text, target_text))
            
        return [{
            'code': code,
            'description': descriptions[code],
            'result': pred_text,
        }]

    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        cur_time = time.time()
        file, phrase = self.preprocess(data)
        print('VP: ', time.time() - cur_time)
        text = self.inference(file)
        return self.postprocess(text, phrase)

    def compare_texts(self, pred_text: str, target_text: str):
        pred_text   = pred_text  .strip().upper()
        target_text = target_text.strip().upper()
        nlev = strsimpy.NormalizedLevenshtein()
        nlev_similarity_score = nlev.similarity(pred_text, target_text)
        return nlev_similarity_score < self.NLEV_THRESHOLD
            