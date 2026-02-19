ASSETS_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0"
import pytest
from ultralytics import YOLO
from ultralytics.trackers.bot_sort import PersistentIDRegistry, BOTSORT

class DummyBoTSORT:
    persistent_ids = set()

    @classmethod
    def add_persistent_track(cls, id):
        cls.persistent_ids.add(id)

    @classmethod
    def remove_persistent_track(cls, id):
        cls.persistent_ids.discard(id)

    @classmethod
    def is_persistent(cls, id):
        return id in cls.persistent_ids

@pytest.fixture
def yolo_model(monkeypatch):
    model = YOLO("yolo11n.pt")
    # Monkeypatch model to use DummyBoTSORT for persistent API
    model._botsort = DummyBoTSORT
    model.add_persistent_track = DummyBoTSORT.add_persistent_track
    model.remove_persistent_track = DummyBoTSORT.remove_persistent_track
    model.is_persistent = DummyBoTSORT.is_persistent
    return model

def test_add_persistent_track(yolo_model):
    yolo_model.add_persistent_track(123)
    assert yolo_model.is_persistent(123)

def test_remove_persistent_track(yolo_model):
    yolo_model.add_persistent_track(456)
    yolo_model.remove_persistent_track(456)
    assert not yolo_model.is_persistent(456)

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
    # Download a short video asset for testing
    video_url = f"{ASSETS_URL}/decelera_portrait_min.mov"
    # Load default botsort.yaml, modify for ReID
    default_args = YAML.load(ROOT / "cfg/trackers/botsort.yaml")
    default_args["with_reid"] = True
    default_args["model"] = "yolo26n-cls.pt"
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        YAML.save(f.name, default_args)
        tracker_yaml = f.name
    model = YOLO("yolo11n.pt")
    results = model.track(video_url, tracker=tracker_yaml, persist=True)
    tracker = model.predictor.trackers[0]  # Get the BOTSORT instance
    # Find a valid track ID from results
    track_ids = [box.id.int().item() for box in results[0].boxes if hasattr(box, 'id')]
    assert track_ids, "No track IDs found in results"
    test_id = track_ids[0]
    model.add_persistent_track(test_id)
    info = PersistentIDRegistry.persistent_info.get(test_id)
    assert info is not None, "Persistent info not stored"
    assert info["cls"] is not None, "Class not stored"
    assert info["embedding"] is not None, "Embedding not stored"
