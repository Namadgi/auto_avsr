import json
import time
import os
from fastapi import UploadFile, HTTPException
import numpy as np
import subprocess
import torch
import asyncio
from typing import Optional, Any, Tuple
from src.models import GCSRequest, ResponseModel
from src.ml_models.vsr import VSR
import re
import strsimpy

VIDEO_OUTPUT = 'output.mp4'

class VSRService:
    NLEV_THRESHOLD = 0.5
    
    def __init__(self):
        self.initialized = False
        self.device = None
        self.model = None
        self.processing_lock = asyncio.Lock()
        self.initialize()

    def initialize(self):
        try:
            self.initialized = True
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = VSR()
            self.model.load_state_dict(torch.load("src/model_weights/vsr.pth"))
            self.model.to(self.device)
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            self.initialized = False
            raise HTTPException(status_code=500, detail="Failed to initialize model")

    async def process_video(self, data: Any) -> ResponseModel:
        """
        Asynchronously process video with lock to ensure sequential processing
        """
        async with self.processing_lock:
            try:
                file_path, phrase = self.preprocess(data)
                text = self.inference(file_path)
                return self.postprocess(text, phrase)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

    def preprocess(self, data: Any) -> Tuple[str, str]:
        """
        Transform raw input into model input data.
        :param data: Input data (either file upload or GCS request)
        :return: tuple of (file_path, phrase)
        """
        if data is None:
            raise HTTPException(status_code=400, detail="Input data is required")

        cur_time = time.time()

        # Clean up existing files
        if os.path.isfile(VIDEO_OUTPUT):
            os.remove(VIDEO_OUTPUT)

        if isinstance(data, UploadFile):
            # Handle file upload
            with open(VIDEO_OUTPUT, 'wb') as out_file:
                content = data.file.read()
                out_file.write(content)
            result_object_name = VIDEO_OUTPUT
            phrase = ""
        else:
            # Handle GCS request
            try:
                gcs_request: GCSRequest = data
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid GCS request format: {str(e)}")

            if not gcs_request.instances or len(gcs_request.instances) == 0:
                raise HTTPException(status_code=400, detail="No instances provided in GCS request")

            instance = gcs_request.instances[0]
            if not instance.token or not instance.bucket_name or not instance.object_name:
                raise HTTPException(status_code=400, detail="Missing required GCS parameters")

            token = instance.token
            bucket_name = instance.bucket_name
            object_name = instance.object_name
            phrase = instance.phrase

            object_encoded_name = object_name.replace('/', '%2F')
            result_object_name = object_name.split('/')[-1]
            curl_cmd = f'curl -X GET -H "Authorization: Bearer {token}" -o {result_object_name} ' + \
                        f'"https://storage.googleapis.com/storage/v1/b/{bucket_name}/o/{object_encoded_name}?alt=media"'
            
            if os.system(curl_cmd) != 0:
                raise HTTPException(status_code=404, detail="Failed to download file from GCS")

        print('DL: ', time.time() - cur_time)

        cur_time = time.time()
        # Process video with ffmpeg
        if result_object_name.lower().endswith('.webm'):
            command = f'ffmpeg -y -fflags +genpts -i {result_object_name} -qscale:v 2 ' + \
                     f'-max_muxing_queue_size 1024 -async 1 -r 25 -vf scale="-2:640" {VIDEO_OUTPUT}'
        else:
            command = f'ffmpeg -y -nostdin -i {result_object_name} -qscale:v 2 ' + \
                     f'-async 1 -r 25 -vf scale="-2:320" {VIDEO_OUTPUT}'

        try:
            subprocess.run(command, shell=True, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            raise HTTPException(status_code=500, detail="Failed to process video with FFmpeg")

        print('VP: ', time.time() - cur_time)

        return VIDEO_OUTPUT, phrase

    def inference(self, file_path: str) -> str:
        """
        Internal inference methods
        :param file_path: Path to the video file
        :return: predicted text
        """
        if not self.initialized:
            raise HTTPException(status_code=503, detail="Model not initialized")

        try:
            self.model.eval()
            with torch.no_grad():
                y = self.model(file_path)
            return y
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    def postprocess(self, pred_text: str, target_text: str) -> ResponseModel:
        """
        Return inference result.
        :param pred_text: Predicted text
        :param target_text: Target text
        :return: ResponseModel
        """
        descriptions = [
            'Successful check',
            'Phrase does not match',
            'No face present',
        ]
        
        similarity = 0
        if pred_text == '':
            code = 2
        else:
            with open('extra_files/word_map.json') as f:
                word_map: dict = json.load(f)
            pred_text = pred_text.strip().upper()
            for pat, repl in word_map.items():
                pred_text = re.sub(pat, repl, pred_text)
            code, similarity = self.compare_texts(pred_text, target_text)
            
        return ResponseModel(
            code=code,
            description=descriptions[code],
            result=pred_text,
            score=round(similarity * 100, 2)
        )

    def compare_texts(self, pred_text: str, target_text: str) -> tuple:
        pred_text = pred_text.strip().upper()
        target_text = target_text.strip().upper()
        nlev = strsimpy.NormalizedLevenshtein()
        nlev_similarity_score = nlev.similarity(pred_text, target_text)
        return int(nlev_similarity_score < self.NLEV_THRESHOLD), nlev_similarity_score
