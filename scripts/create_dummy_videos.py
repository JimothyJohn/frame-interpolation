import subprocess
import os

resolutions = {
    "480p": "854x480",
    "720p": "1280x720",
    "1080p": "1920x1080"
}
framerates = [15, 30]
lengths = [0.25, 0.5]

output_dir = "dummy_videos"
os.makedirs(output_dir, exist_ok=True)

def create_video(res_name, width_height, fps, length):
    filename = f"dummy_{res_name}_{fps}fps_{length}s.mp4"
    filepath = os.path.join(output_dir, filename)
    
    # Using testsrc to generate a test pattern video with a timestamp
    # -y overwrites output
    # -f lavfi -i testsrc... generates the source
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"testsrc=size={width_height}:rate={fps}:duration={length}",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        filepath
    ]
    
    print(f"Generating {filename}...")
    try:
        subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)
        print(f"Successfully created {filepath}")
    except subprocess.CalledProcessError as e:
        print(f"Error creating {filepath}: {e}")
    except FileNotFoundError:
        print("ffmpeg not found. Please install ffmpeg.")
        return

def main():
    for res_name, size in resolutions.items():
        for fps in framerates:
            for length in lengths:
                create_video(res_name, size, fps, length)

if __name__ == "__main__":
    main()

