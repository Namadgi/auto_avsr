import json
import time
import os
from fastapi import UploadFile, HTTPException
import numpy as np
import subprocess
import torch
import asyncio
from typing import Optional, Any, Tuple, List
from src.models import GCSRequest, ResponseModel
from src.ml_models.vsr import VSR
import re
import strsimpy

# OpenTelemetry imports for GCP
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
from opentelemetry.exporter.cloud_monitoring import CloudMonitoringMetricsExporter
from opentelemetry.trace import Status, StatusCode

VIDEO_OUTPUT = 'vsr_output.mp4'
PROJECT_ID = "biometry-416410"

class VSRService:
    NLEV_THRESHOLD = 0.5

    def __init__(self):
        self.initialized = False
        self.device = None
        self.model = None
        # self.processing_lock = asyncio.Lock()

        # Initialize OpenTelemetry
        self._setup_telemetry()
        self.initialize()

    def _setup_telemetry(self):
        """Initialize OpenTelemetry tracer and meter for GCP."""
        # Set up tracing for GCP Cloud Trace
        trace.set_tracer_provider(TracerProvider())
        tracer_provider = trace.get_tracer_provider()
        tracer_provider.add_span_processor(
            BatchSpanProcessor(CloudTraceSpanExporter(project_id=PROJECT_ID))
        )
        self.tracer = trace.get_tracer(__name__)

        # Set up metrics for GCP Cloud Monitoring
        reader = PeriodicExportingMetricReader(
            exporter=CloudMonitoringMetricsExporter(project_id=PROJECT_ID),
            export_interval_millis=60000
        )
        metrics.set_meter_provider(MeterProvider(metric_readers=[reader]))
        self.meter = metrics.get_meter(__name__)

        # Create metrics
        self.request_counter = self.meter.create_counter(
            name="vsr_requests_total",
            description="Total number of VSR requests",
            unit="1"
        )
        self.request_duration = self.meter.create_histogram(
            name="vsr_request_duration_seconds",
            description="Duration of VSR requests",
            unit="s"
        )
        self.preprocess_duration = self.meter.create_histogram(
            name="vsr_preprocess_duration_seconds",
            description="Duration of preprocessing operations",
            unit="s"
        )
        self.inference_duration = self.meter.create_histogram(
            name="vsr_inference_duration_seconds",
            description="Duration of inference operations",
            unit="s"
        )
        self.postprocess_duration = self.meter.create_histogram(
            name="vsr_postprocess_duration_seconds",
            description="Duration of postprocessing operations",
            unit="s"
        )

    def initialize(self):
        try:
            with self.tracer.start_as_current_span("model_initialization") as span:
                span.set_attribute("device.type", "cuda" if torch.cuda.is_available() else "cpu")

                self.initialized = True
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model = VSR()
                self.model.load_state_dict(torch.load("src/model_weights/vsr.pth"))
                self.model.to(self.device)

                span.set_status(Status(StatusCode.OK))
                span.set_attribute("model.loaded", True)
        except Exception as e:
            with self.tracer.start_as_current_span("model_initialization") as span:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.set_attribute("model.loaded", False)
            self.initialized = False
            raise HTTPException(status_code=500, detail="Failed to initialize model")

    async def process_video(self, data: Any) -> ResponseModel:
        """
        Asynchronously process video with telemetry instrumentation
        """
        # async with self.processing_lock:
        with self.tracer.start_as_current_span("vsr_process_video") as span:
            try:
                # Set request attributes
                span.set_attribute("service.name", "vsr_service")
                span.set_attribute("operation", "process_video")

                overall_start_time = time.time()

                # Preprocess with telemetry
                with self.tracer.start_as_current_span("preprocess") as preprocess_span:
                    step_start_time = time.time()
                    file_path, phrase, preprocess_timings = self.preprocess(data)
                    preprocess_duration = time.time() - step_start_time

                    # Record preprocess metrics and attributes
                    self.preprocess_duration.record(preprocess_duration)
                    preprocess_span.set_attribute("duration_seconds", preprocess_duration)
                    preprocess_span.set_attribute("file_path", file_path)
                    preprocess_span.set_attribute("phrase", phrase)

                    # Record sub-operation timings
                    for sub_name, sub_dur in preprocess_timings:
                        preprocess_span.set_attribute(f"sub_operation.{sub_name}", sub_dur)

                # Inference with telemetry
                with self.tracer.start_as_current_span("inference") as inference_span:
                    step_start_time = time.time()
                    text = self.inference(file_path)
                    inference_duration = time.time() - step_start_time

                    # Record inference metrics and attributes
                    self.inference_duration.record(inference_duration)
                    inference_span.set_attribute("duration_seconds", inference_duration)
                    inference_span.set_attribute("predicted_text", text)

                # Postprocess with telemetry
                with self.tracer.start_as_current_span("postprocess") as postprocess_span:
                    step_start_time = time.time()
                    response = self.postprocess(text, phrase)
                    postprocess_duration = time.time() - step_start_time

                    # Record postprocess metrics and attributes
                    self.postprocess_duration.record(postprocess_duration)
                    postprocess_span.set_attribute("duration_seconds", postprocess_duration)
                    postprocess_span.set_attribute("response.code", response.code)
                    postprocess_span.set_attribute("response.score", response.score)

                overall_duration = time.time() - overall_start_time

                # Record overall metrics
                self.request_duration.record(overall_duration, {
                    "status": "success",
                    "response_code": response.code
                })
                self.request_counter.add(1, {
                    "status": "success",
                    "response_code": response.code
                })

                # Set span attributes
                span.set_attribute("overall_duration_seconds", overall_duration)
                span.set_attribute("response_code", response.code)
                span.set_attribute("response_description", response.description)
                span.set_attribute("similarity_score", response.score)

                span.set_status(Status(StatusCode.OK))
                return response

            except Exception as e:
                # Record failure metrics
                overall_duration = time.time() - overall_start_time
                self.request_duration.record(overall_duration, {
                    "status": "error",
                    "response_code": 500
                })
                self.request_counter.add(1, {"status": "error"})

                # Record exception in span
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.set_attribute("error", True)
                span.set_attribute("error_message", str(e))

                raise HTTPException(status_code=500, detail=str(e))

    def preprocess(self, data: Any) -> Tuple[str, str, List[Tuple[str, float]]]:
        """
        Transform raw input into model input data.
        :param data: Input data (either file upload or GCS request)
        :return: tuple of (file_path, phrase, timings)
        """
        if data is None:
            raise HTTPException(status_code=400, detail="Input data is required")

        timings: List[Tuple[str, float]] = []

        # Clean up existing files
        if os.path.isfile(VIDEO_OUTPUT):
            os.remove(VIDEO_OUTPUT)

        if isinstance(data, UploadFile):
            # Handle file upload
            with self.tracer.start_as_current_span("upload_file_handling") as span:
                span.set_attribute("input_type", "upload_file")
                span.set_attribute("file_name", getattr(data, 'filename', 'unknown'))

                start_ts = time.time()
                with open(VIDEO_OUTPUT, 'wb') as out_file:
                    content = data.file.read()
                    out_file.write(content)
                duration = time.time() - start_ts
                timings.append(("save_upload", duration))

                span.set_attribute("upload_duration_seconds", duration)
                span.set_attribute("file_size_bytes", len(content))

            result_object_name = VIDEO_OUTPUT
            phrase = ""
        else:
            # Handle GCS request
            with self.tracer.start_as_current_span("gcs_request_handling") as span:
                span.set_attribute("input_type", "gcs_request")

                try:
                    gcs_request: GCSRequest = data
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, "Invalid GCS request format"))
                    raise HTTPException(status_code=400, detail=f"Invalid GCS request format: {str(e)}")

                if not gcs_request.instances or len(gcs_request.instances) == 0:
                    span.set_status(Status(StatusCode.ERROR, "No instances provided"))
                    raise HTTPException(status_code=400, detail="No instances provided in GCS request")

                instance = gcs_request.instances[0]
                if not instance.token or not instance.bucket_name or not instance.object_name:
                    span.set_status(Status(StatusCode.ERROR, "Missing GCS parameters"))
                    raise HTTPException(status_code=400, detail="Missing required GCS parameters")

                token = instance.token
                bucket_name = instance.bucket_name
                object_name = instance.object_name
                phrase = instance.phrase

                span.set_attribute("gcs.bucket_name", bucket_name)
                span.set_attribute("gcs.object_name", object_name)

                object_encoded_name = object_name.replace('/', '%2F')
                result_object_name = f"vsr_{object_name.split('/')[-1]}"

                with self.tracer.start_as_current_span("gcs_download") as download_span:
                    curl_cmd = f'curl -X GET -H "Authorization: Bearer {token}" -o {result_object_name} ' + \
                                f'"https://storage.googleapis.com/download/storage/v1/b/{bucket_name}/o/{object_encoded_name}?alt=media"'

                    start_ts = time.time()
                    exit_code = os.system(curl_cmd)
                    duration = time.time() - start_ts

                    if exit_code != 0:
                        download_span.set_status(Status(StatusCode.ERROR, "GCS download failed"))
                        download_span.set_attribute("curl_exit_code", exit_code)
                        raise HTTPException(status_code=404, detail="Failed to download file from GCS")

                    timings.append(("download_gcs", duration))
                    download_span.set_attribute("download_duration_seconds", duration)
                    download_span.set_attribute("download_success", True)

        # Process video with ffmpeg
        with self.tracer.start_as_current_span("ffmpeg_transcoding") as ffmpeg_span:
            ffmpeg_span.set_attribute("input_file", result_object_name)
            ffmpeg_span.set_attribute("output_file", VIDEO_OUTPUT)

            if result_object_name.lower().endswith('.webm'):
                command = f'ffmpeg -y -fflags +genpts -i {result_object_name} -qscale:v 2 ' + \
                         f'-max_muxing_queue_size 1024 -async 1 -r 25 -vf scale="-2:640" {VIDEO_OUTPUT}'
                ffmpeg_span.set_attribute("input_format", "webm")
                ffmpeg_span.set_attribute("scale_resolution", "640")
            else:
                command = f'ffmpeg -y -nostdin -i {result_object_name} -qscale:v 2 ' + \
                         f'-async 1 -r 25 -vf scale="-2:320" {VIDEO_OUTPUT}'
                ffmpeg_span.set_attribute("input_format", "other")
                ffmpeg_span.set_attribute("scale_resolution", "320")

            try:
                start_ts = time.time()
                subprocess.run(command, shell=True, check=True, capture_output=True)
                duration = time.time() - start_ts
                timings.append(("ffmpeg_transcode", duration))

                ffmpeg_span.set_attribute("transcode_duration_seconds", duration)
                ffmpeg_span.set_attribute("transcode_success", True)

            except subprocess.CalledProcessError as e:
                ffmpeg_span.record_exception(e)
                ffmpeg_span.set_status(Status(StatusCode.ERROR, "FFmpeg transcoding failed"))
                ffmpeg_span.set_attribute("ffmpeg_command", command)
                raise HTTPException(status_code=500, detail="Failed to process video with FFmpeg")

        return VIDEO_OUTPUT, phrase, timings

    def inference(self, file_path: str) -> str:
        """
        Internal inference methods
        :param file_path: Path to the video file
        :return: predicted text
        """
        if not self.initialized:
            raise HTTPException(status_code=503, detail="Model not initialized")

        with self.tracer.start_as_current_span("model_inference") as span:
            span.set_attribute("file_path", file_path)
            span.set_attribute("device", str(self.device))

            try:
                self.model.eval()

                with self.tracer.start_as_current_span("torch_inference") as torch_span:
                    with torch.no_grad():
                        y = self.model(file_path)

                span.set_attribute("predicted_text", str(y))
                span.set_attribute("prediction_success", True)
                span.set_status(Status(StatusCode.OK))

                return y

            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, f"Inference error: {str(e)}"))
                span.set_attribute("prediction_success", False)
                raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    def postprocess(self, pred_text: str, target_text: str) -> ResponseModel:
        """
        Return inference result.
        :param pred_text: Predicted text
        :param target_text: Target text
        :return: ResponseModel
        """
        with self.tracer.start_as_current_span("postprocess_text") as span:
            span.set_attribute("predicted_text", pred_text)
            span.set_attribute("target_text", target_text)

            descriptions = [
                'Successful check',
                'Phrase does not match',
                'No face present',
            ]

            similarity = 0
            if pred_text == '':
                code = 2
                span.set_attribute("processing_result", "no_face_detected")
            else:
                with self.tracer.start_as_current_span("text_processing") as text_span:
                    with open('extra_files/word_map.json') as f:
                        word_map: dict = json.load(f)

                    original_pred_text = pred_text
                    pred_text = pred_text.strip().upper()

                    # Apply word mapping transformations
                    transformations_applied = []
                    for pat, repl in word_map.items():
                        if re.search(pat, pred_text):
                            pred_text = re.sub(pat, repl, pred_text)
                            transformations_applied.append(f"{pat}->{repl}")

                    text_span.set_attribute("original_text", original_pred_text)
                    text_span.set_attribute("processed_text", pred_text)
                    text_span.set_attribute("transformations_applied", str(transformations_applied))

                code, similarity = self.compare_texts(pred_text, target_text)
                span.set_attribute("processing_result", "text_comparison_completed")

            span.set_attribute("response_code", code)
            span.set_attribute("response_description", descriptions[code])
            span.set_attribute("similarity_score", round(similarity * 100, 2))
            span.set_attribute("similarity_percentage", round(similarity * 100, 2))

            response = ResponseModel(
                code=code,
                description=descriptions[code],
                result=pred_text,
                score=round(similarity * 100, 2)
            )

            span.set_status(Status(StatusCode.OK))
            return response

    def compare_texts(self, pred_text: str, target_text: str) -> tuple:
        with self.tracer.start_as_current_span("text_similarity_comparison") as span:
            span.set_attribute("predicted_text_clean", pred_text.strip().upper())
            span.set_attribute("target_text_clean", target_text.strip().upper())

            pred_text = pred_text.strip().upper()
            target_text = target_text.strip().upper()

            nlev = strsimpy.NormalizedLevenshtein()
            nlev_similarity_score = nlev.similarity(pred_text, target_text)

            span.set_attribute("similarity_algorithm", "normalized_levenshtein")
            span.set_attribute("similarity_score", nlev_similarity_score)
            span.set_attribute("similarity_threshold", self.NLEV_THRESHOLD)
            span.set_attribute("similarity_passed", nlev_similarity_score >= self.NLEV_THRESHOLD)

            span.set_status(Status(StatusCode.OK))
            return int(nlev_similarity_score < self.NLEV_THRESHOLD), nlev_similarity_score
