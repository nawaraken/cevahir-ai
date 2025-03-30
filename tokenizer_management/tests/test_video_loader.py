import cv2
import numpy as np
from pathlib import Path
import pytest
from tokenizer_management.data_loader.video_loader import VideoLoader

def create_dummy_video(file_path: Path, frame_count=20, frame_size=(320, 240), fps=10):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(file_path), fourcc, fps, frame_size)
    for _ in range(frame_count):
        frame = np.random.randint(0, 256, (frame_size[1], frame_size[0], 3), dtype=np.uint8)
        out.write(frame)
    out.release()

def test_video_loader(tmp_path: Path):
    file = tmp_path / "test.mp4"
    create_dummy_video(file, frame_count=20)
    
    loader = VideoLoader(desired_frames=8, resize=(112, 112))
    output = loader.load_file(str(file))
    # Çıktı, (8, 112, 112, 3) boyutunda bir numpy array olmalıdır.
    assert isinstance(output, np.ndarray)
    assert output.shape[0] == 8
    assert output.shape[1] == 112
    assert output.shape[2] == 112
    assert output.shape[3] == 3
