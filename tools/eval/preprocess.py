"""Mirror of the ffmpeg transcode step in VSRService.preprocess, with per-call temp paths.
Production writes a fixed 'vsr_output.mp4'; the commands below are otherwise identical."""
import os
import subprocess
import tempfile


def transcode(src: str, out_dir: str = None) -> str:
    """Return path of the 25 fps mp4 the model actually sees for `src`."""
    fd, dst = tempfile.mkstemp(suffix='.mp4', prefix='vsr_', dir=out_dir)
    os.close(fd)
    if src.lower().endswith('.webm'):
        cmd = ['ffmpeg', '-y', '-fflags', '+genpts', '-i', src, '-qscale:v', '2',
               '-max_muxing_queue_size', '1024', '-async', '1', '-r', '25', '-vf', 'scale=-2:640', dst]
    else:
        cmd = ['ffmpeg', '-y', '-nostdin', '-i', src, '-qscale:v', '2', '-async', '1', '-r', '25', '-vf', 'scale=-2:320', dst]
    subprocess.run(cmd, check=True, capture_output=True)
    return dst


def extract_audio(src: str, out_dir: str = None) -> str:
    """16 kHz mono wav for ASR labeling."""
    fd, dst = tempfile.mkstemp(suffix='.wav', prefix='vsr_audio_', dir=out_dir)
    os.close(fd)
    subprocess.run(['ffmpeg', '-y', '-nostdin', '-i', src, '-vn', '-ac', '1', '-ar', '16000', dst],
                   check=True, capture_output=True)
    return dst
