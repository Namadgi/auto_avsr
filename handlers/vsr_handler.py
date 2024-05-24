import os
import re
import numpy as np
import json

import torch
from ts.torch_handler.base_handler import BaseHandler

import subprocess
import time
import strsimpy

SCALE = 0.25
# VIDEO_TEMP   = 'temp.avi'
VIDEO_OUTPUT = 'output.mp4'

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

        # if os.path.isfile(VIDEO_TEMP):
        #     os.remove(VIDEO_TEMP)
        
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
                with open(VIDEO_OUTPUT, 'wb') as out_file:
                    out_file.write(data)
                    object_name = VIDEO_OUTPUT
        phrase = ''
        if is_dict:
            # Download file
            token = data['token']
            bucket_name = data['bucket_name']
            object_name = data['object_name']
            phrase = data['phrase']
            object_encoded_name = object_name.replace('/', '%2F')
            result_object_name = object_name.split('/')[-1]
            print(data)

            os.system(
                f'curl -X GET ' +
                f'-H "Authorization: Bearer {token}" -o {result_object_name} '
                f'"https://storage.googleapis.com/storage/v1/b/{bucket_name}/o/{object_encoded_name}?alt=media"'
            )

        new_res_object_name = VIDEO_OUTPUT

        if result_object_name[-5:] == '.webm':
            command = f'ffmpeg -y -fflags +genpts -i {result_object_name} ' +\
                f'-max_muxing_queue_size 1024 -r 25 {new_res_object_name}'
            os.system(command)
        else:
            new_res_object_name = result_object_name

        subprocess.call(
            f'ffmpeg -y -i {new_res_object_name} -qscale:v 2 -threads 10 ' +\
            f'-async 1 -r 25 -vf scale="-2:640" {VIDEO_OUTPUT} -loglevel panic',
            shell=True, 
            stdout=None,    
        )

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
            with open('word_map.json') as f:
                word_map = json.load(f)
            pred_text = pred_text.strip().upper()
            for pat, repl in word_map.items():
                pred_text = re.sub(pat, repl, pred_text)
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
        result = self.postprocess(text, phrase)
        print(result)
        return result

    def compare_texts(self, pred_text: str, target_text: str):
        pred_text   = pred_text  .strip().upper()
        target_text = target_text.strip().upper()
        nlev = strsimpy.NormalizedLevenshtein()
        nlev_similarity_score = nlev.similarity(pred_text, target_text)
        return nlev_similarity_score < self.NLEV_THRESHOLD
            