import pytest
import os

@pytest.fixture
def base_url():
    server_url = os.getenv("SERVER_URL", "http://localhost")
    server_port = os.getenv("SERVER_PORT", "8080")
    return f"{server_url}:{server_port}"
