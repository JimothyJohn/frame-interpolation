import requests
import pytest

def test_root_endpoint(base_url):
    """
    Verifies the endpoint provides the proper API specification at the root path of the server.
    Expected response:
    {
        "cog_version": "0.16.9",
        "docs_url": "/docs",
        "openapi_url": "/openapi.json",
        "shutdown_url": "/shutdown",
        "healthcheck_url": "/health-check",
        "predictions_url": "/predictions",
        "predictions_idempotent_url": "/predictions/{prediction_id}",
        "predictions_cancel_url": "/predictions/{prediction_id}/cancel"
    }
    """
    response = requests.get(f"{base_url}/")
    assert response.status_code == 200
    
    data = response.json()
    
    # Verify specific keys and values as requested
    assert data["cog_version"] == "0.16.9"
    assert data["docs_url"] == "/docs"
    assert data["openapi_url"] == "/openapi.json"
    assert data["shutdown_url"] == "/shutdown"
    assert data["healthcheck_url"] == "/health-check"
    assert data["predictions_url"] == "/predictions"
    assert data["predictions_idempotent_url"] == "/predictions/{prediction_id}"
    assert data["predictions_cancel_url"] == "/predictions/{prediction_id}/cancel"

def test_prediction(base_url):
    """
    Verifies the /predictions endpoint works as expected.
    """
    import base64
    import time
    import logging
    
    logger = logging.getLogger(__name__)
    
    # Read the dummy video file
    video_path = "tests/dummy_videos/dummy_480p_15fps_2s.mp4"
    with open(video_path, "rb") as f:
        video_data = f.read()
        video_base64 = base64.b64encode(video_data).decode("utf-8")
        video_data_uri = f"data:video/mp4;base64,{video_base64}"

    # Create prediction
    payload = {
        "input": {
            "video": video_data_uri,
            "target_fps": 30,
            "slowdown": 2.0
        }
    }
    
    response = requests.post(f"{base_url}/predictions", json=payload)
    assert response.status_code in [200, 201]
    prediction = response.json()
    logger.info(f"Prediction response: {prediction}")
    
    prediction_id = prediction.get("id")
    if not prediction_id:
        # Handle case where server returns a log-like response indicating success
        if prediction.get("message") == "prediction succeeded":
            logger.info("Prediction succeeded immediately (synchronous/log response)")
            return
        elif prediction.get("status") == "succeeded":
             logger.info("Prediction succeeded immediately")
             return
        else:
            pytest.fail(f"Prediction failed or no ID returned: {prediction}")

    # Poll for completion
    start_time = time.time()
    while time.time() - start_time < 300:  # 5 minute timeout
        response = requests.get(f"{base_url}/predictions/{prediction_id}")
        logger.debug(f"Requesting {response.url}")
        logger.debug(f"Status {response.status_code}")
        logger.debug(f"Body {response.text}")
        assert response.status_code == 200
        prediction = response.json()
        
        status = prediction["status"]
        if status == "succeeded":
            break
        elif status == "failed":
            pytest.fail(f"Prediction failed: {prediction.get('error')}")
            
        time.sleep(1)
    else:
        pytest.fail("Prediction timed out")
        
    # Verify output
    assert "output" in prediction
    assert prediction["output"] is not None
