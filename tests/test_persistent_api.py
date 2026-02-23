ASSETS_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0"
import pytest
from ultralytics import YOLO
from ultralytics.trackers.bot_sort import BOTSORT

@pytest.fixture
def yolo_model():
    model = YOLO("yolo11n.pt")
    return model

def test_add_persistent_track(yolo_model):
    yolo_model.add_persistent_track(123)
    assert yolo_model.is_persistent(123)
    assert 123 in yolo_model._persistent_ids

def test_remove_persistent_track(yolo_model):
    yolo_model.add_persistent_track(456)
    yolo_model.remove_persistent_track(456)
    assert not yolo_model.is_persistent(456)
    assert 456 not in yolo_model._persistent_ids

def test_multiple_persistent_tracks(yolo_model):
    ids = [1, 2, 3]
    for i in ids:
        yolo_model.add_persistent_track(i)
    for i in ids:
        assert yolo_model.is_persistent(i)
    yolo_model.remove_persistent_track(2)
    assert not yolo_model.is_persistent(2)
    assert yolo_model.is_persistent(1)
    assert yolo_model.is_persistent(3)

def test_persistent_id_does_not_affect_non_persistent(yolo_model):
    yolo_model.add_persistent_track(999)
    assert not yolo_model.is_persistent(1000)

# Integration test for persistent track info
def test_persistent_track_registry():
    import tempfile
    from ultralytics.utils import YAML, ROOT
    from pathlib import Path
    
    # Use a local tiny image instead of downloading video for quick test if possible,
    # or keep video URL. Since track IDs are needed, we trace a video list.
    video_url = "https://ultralytics.com/images/bus.jpg" # dummy, just to trigger tracking
    
    # Load default botsort.yaml
    default_args = YAML.load(ROOT / "cfg/trackers/botsort.yaml")
    
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        YAML.save(f.name, default_args)
        tracker_yaml = f.name
        
    model = YOLO("yolo11n.pt")
    model.add_persistent_track(1) # Test queueing before tracker exists!
    
    results = model.track(video_url, tracker=tracker_yaml, persist=True)
    tracker = model.predictor.trackers[0]  # Get the BOTSORT instance
    
    # Assert queued ID made it to the tracker
    assert tracker.is_persistent(1), "Queued persistent ID was not passed to the tracker"

    # Find a valid track ID from results
    track_ids = [int(box.id.item()) for r in results for box in r.boxes if box.id is not None]
    if track_ids:
        test_id = track_ids[0]
        model.add_persistent_track(test_id)
        
        info = tracker.persistent_info.get(test_id, {})
        assert info, "Persistent info not stored in tracker instance"
        # We can't guarantee embedding/cls here without reid model, but it's okay, 
        # just ensuring the info dict exists at instance level.
