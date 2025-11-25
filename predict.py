import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TF info/warning logs
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from pathlib import Path
import numpy as np
import tempfile
import tensorflow as tf
import cog
from cog import BasePredictor, Input, Path
import shutil
import subprocess
import math
import logging
from tqdm import tqdm
import time
import sys

from eval import interpolator, util

# Configure logging
logging.basicConfig(filename='prediction.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

_UINT8_MAX_F = float(np.iinfo(np.uint8).max)

class Predictor(BasePredictor):
    def setup(self):
        import tensorflow as tf
        try:
            # Attempt to configure GPU memory growth
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    except RuntimeError as e:
                         # Visible devices must be set before GPUs have been initialized
                         logging.warning(f"Could not set memory growth: {e}")
                         pass
                logging.info(f"GPU(s) detected: {len(gpus)}")
            else:
                logging.info("Running on CPU")
        except Exception as e:
             logging.error(f"Error setting up GPU memory growth: {e}")

        try:
            self.interpolator = interpolator.Interpolator("pretrained_models/film_net/Style/saved_model", None)
            logging.info("Model loaded successfully")
        except Exception as e:
            logging.critical(f"Failed to load interpolator: {e}")
            # Re-raise to fail setup explicitly, but log it first
            raise e

        # Batched time.
        self.batch_dt = np.full(shape=(1,), fill_value=0.5, dtype=np.float32)


    def predict(
        self,
        video: Path = Input(description="Input video file"),
        target_fps: int = Input(description="Target FPS for the output video", default=60),
        slowdown: float = Input(description="Slowdown factor (e.g., 2.0 means half speed). Default 1.0", default=1.0, ge=1.0, le=10.0)
    ) -> Path:
        
        # Check if video exists
        if not os.path.exists(str(video)):
            raise ValueError("Input video not found.")
        
        # Create temp directories
        temp_dir = Path(tempfile.mkdtemp())
        input_frames_dir = temp_dir / "input_frames"
        output_frames_dir = temp_dir / "output_frames"
        input_frames_dir.mkdir(parents=True, exist_ok=True)
        output_frames_dir.mkdir(parents=True, exist_ok=True)
        
        ffmpeg_path = util.get_ffmpeg_path()

        # Get FPS
        fps = self._get_fps(video)
        logging.info(f"Input FPS: {fps}")
        
        if fps == 0:
            fps = 30.0
            logging.warning("Could not determine FPS, defaulting to 30.0")

        effective_target_fps = target_fps * slowdown
        
        times_to_interpolate = 0
        if effective_target_fps > fps:
             times_to_interpolate = math.ceil(math.log2(effective_target_fps / fps))
        
        if times_to_interpolate < 0:
            times_to_interpolate = 0
            
        logging.info(f"Slowdown: {slowdown}, Effective Target FPS: {effective_target_fps}")
        logging.info(f"Times to interpolate: {times_to_interpolate}")
        
        # Extract frames
        logging.info("Extracting frames...")
        cmd = [
            ffmpeg_path, '-i', str(video),
            '-vsync', '0', 
            str(input_frames_dir / '%06d.png')
        ]
        subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)
        
        input_frames = sorted([str(p) for p in input_frames_dir.glob('*.png')])
        if not input_frames:
            raise ValueError("No frames extracted from video.")
            
        logging.info(f"Extracted {len(input_frames)} frames.")

        # Interpolate
        logging.info("Interpolating...")
        interpolated_frames = util.interpolate_recursively_from_files(
            input_frames, times_to_interpolate, self.interpolator
        )
        
        # Save interpolated frames
        i = 0
        
        # Standard tqdm configuration to avoid newline spam
        pbar = tqdm(interpolated_frames, desc="Interpolating", unit="frame")
                   
        for frame in pbar:
            util.write_image(str(output_frames_dir / f"{i:06d}.png"), frame)
            i += 1
            
        logging.info(f"Saved {i} output frames.")
        
        # Create output video
        out_path = Path(tempfile.mkdtemp()) / "out.mp4"
        
        logging.info(f"Encoding video at {target_fps} FPS...")

        # Use ffmpeg to stitch
        cmd = [
            ffmpeg_path,
            '-r', str(target_fps),
            '-i', str(output_frames_dir / '%06d.png'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-y',
            str(out_path)
        ]
        subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)
        
        # Clean up temp dir
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        logging.info(f"Processing complete. Saved to: {out_path}")
        
        return out_path

    def _get_fps(self, video_path):
        cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=r_frame_rate',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(video_path)
        ]
        try:
            output = subprocess.check_output(cmd).decode('utf-8').strip()
            if '/' in output:
                num, den = output.split('/')
                if float(den) == 0: return 0.0
                return float(num) / float(den)
            return float(output)
        except Exception as e:
            logging.error(f"Error getting FPS: {e}")
            return 0.0
