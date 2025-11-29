import os
import pytest
from predict import Predictor
from cog import Path

@pytest.fixture
def predictor():
    p = Predictor()
    p.setup()
    return p

def test_predictor_setup(predictor):
    """Verifies that the predictor is set up correctly."""
    assert hasattr(predictor, "interpolator")
    assert predictor.interpolator is not None

def dont_test_predictor_predict(predictor):
    """Verifies that the predictor can run a prediction."""
    video_path = "tests/dummy_videos/dummy_480p_15fps_1s.mp4"
    assert os.path.exists(video_path), f"Video file not found: {video_path}"
    
    output_path = predictor.predict(
        video=Path(video_path),
        target_fps=30,
        slowdown=2.0
    )
    
    assert output_path is not None
    assert os.path.exists(str(output_path))
    assert str(output_path).endswith(".mp4")
