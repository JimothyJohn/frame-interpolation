import pytest
import time
import json
import os
from predict import Predictor
from cog import Path

@pytest.fixture(scope="module")
def predictor():
    p = Predictor()
    p.setup()
    return p

def test_performance(predictor):
    results = []
    
    test_cases = [
        {"name": "480p_15fps", "file": "tests/dummy_videos/dummy_480p_15fps_1s.mp4"},
        {"name": "480p_30fps", "file": "tests/dummy_videos/dummy_480p_30fps_1s.mp4"},
        {"name": "1080p_15fps", "file": "tests/dummy_videos/dummy_1080p_15fps_1s.mp4"},
        {"name": "1080p_30fps", "file": "tests/dummy_videos/dummy_1080p_30fps_1s.mp4"},
    ]
    
    for case in test_cases:
        video_path = case["file"]
        assert os.path.exists(video_path), f"File not found: {video_path}"
        
        print(f"Running performance test for {case['name']}...")
        start_time = time.time()
        
        output_path = predictor.predict(
            video=Path(video_path),
            target_fps=60, # Interpolating to 60fps as a standard target
            slowdown=1.0
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        results.append({
            "name": case["name"],
            "duration_seconds": duration,
            "input_file": video_path
        })
        print(f"Finished {case['name']} in {duration:.2f}s")

    output_file = "tests/performance_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Performance results saved to {output_file}")
    assert len(results) == len(test_cases)
