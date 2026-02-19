import pytest
from ultralytics import YOLO

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
