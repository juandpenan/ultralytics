# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations
from ultralytics.utils import LOGGER
from collections import deque

# Persistent ID registry for BOTSORT
class PersistentIDRegistry:
    persistent_info = {}

    @classmethod
    def add_persistent_track(cls, id, tracker=None):
        id = int(id)
        if id not in cls.persistent_info:
            cls.persistent_info[id] = {"cls": None, "embedding": None}
        # If tracker is provided, try to capture info immediately
        if tracker is not None:
            tracked = getattr(tracker, 'tracked_stracks', [])
            for track in tracked:
                tid = getattr(track, 'track_id', None)
                if tid == id:
                    cls_id = getattr(track, 'cls', None)
                    embedding = getattr(track, 'smooth_feat', None)
                    cls.persistent_info[id] = {"cls": cls_id, "embedding": embedding}
                    break

    @classmethod
    def remove_persistent_track(cls, id):
        cls.persistent_info.pop(int(id), None)

    @classmethod
    def is_persistent(cls, id):
        return int(id) in cls.persistent_info
from typing import Any

import numpy as np
import torch

from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.plotting import save_one_box

from .basetrack import BaseTrack, TrackState
from .byte_tracker import BYTETracker, STrack
from .utils import matching
from .utils.gmc import GMC
from .utils.kalman_filter import KalmanFilterXYWH


class BOTrack(STrack):
    """An extended version of the STrack class for YOLO, adding object tracking features.

    This class extends the STrack class to include additional functionalities for object tracking, such as feature
    smoothing, Kalman filter prediction, and reactivation of tracks.

    Attributes:
        shared_kalman (KalmanFilterXYWH): A shared Kalman filter for all instances of BOTrack.
        smooth_feat (np.ndarray): Smoothed feature vector.
        curr_feat (np.ndarray): Current feature vector.
        features (deque): A deque to store feature vectors with a maximum length defined by `feat_history`.
        alpha (float): Smoothing factor for the exponential moving average of features.
        mean (np.ndarray): The mean state of the Kalman filter.
        covariance (np.ndarray): The covariance matrix of the Kalman filter.

    Methods:
        update_features: Update features vector and smooth it using exponential moving average.
        predict: Predict the mean and covariance using Kalman filter.
        re_activate: Reactivate a track with updated features and optionally new ID.
        update: Update the track with new detection and frame ID.
        tlwh: Property that gets the current position in tlwh format `(top left x, top left y, width, height)`.
        multi_predict: Predict the mean and covariance of multiple object tracks using shared Kalman filter.
        convert_coords: Convert tlwh bounding box coordinates to xywh format.
        tlwh_to_xywh: Convert bounding box to xywh format `(center x, center y, width, height)`.

    Examples:
        Create a BOTrack instance and update its features
        >>> bo_track = BOTrack(xywh=np.array([100, 50, 80, 40, 0]), score=0.9, cls=1, feat=np.random.rand(128))
        >>> bo_track.predict()
        >>> new_track = BOTrack(xywh=np.array([110, 60, 80, 40, 0]), score=0.85, cls=1, feat=np.random.rand(128))
        >>> bo_track.update(new_track, frame_id=2)
    """

    shared_kalman = KalmanFilterXYWH()

    def __init__(
        self, xywh: np.ndarray, score: float, cls: int, feat: np.ndarray | None = None, feat_history: int = 50
    ):
        """Initialize a BOTrack object with temporal parameters, such as feature history, alpha, and current features.

        Args:
            xywh (np.ndarray): Bounding box in `(x, y, w, h, idx)` or `(x, y, w, h, angle, idx)` format, where (x, y) is
                the center, (w, h) are width and height, and `idx` is the detection index.
            score (float): Confidence score of the detection.
            cls (int): Class ID of the detected object.
            feat (np.ndarray, optional): Feature vector associated with the detection.
            feat_history (int): Maximum length of the feature history deque.
        """
        super().__init__(xywh, score, cls)

        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)
        self.features = deque(maxlen=feat_history)
        self.alpha = 0.9

    def update_features(self, feat: np.ndarray) -> None:
        """Update the feature vector and apply exponential moving average smoothing."""
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self) -> None:
        """Predict the object's future state using the Kalman filter to update its mean and covariance."""
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    def re_activate(self, new_track: BOTrack, frame_id: int, new_id: bool = False) -> None:
        """Reactivate a track with updated features and optionally assign a new ID."""
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        super().re_activate(new_track, frame_id, new_id)

    def update(self, new_track: BOTrack, frame_id: int) -> None:
        """Update the track with new detection information and the current frame ID."""
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        super().update(new_track, frame_id)

    @property
    def tlwh(self) -> np.ndarray:
        """Return the current bounding box position in `(top left x, top left y, width, height)` format."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @staticmethod
    def multi_predict(stracks: list[BOTrack]) -> None:
        """Predict the mean and covariance for multiple object tracks using a shared Kalman filter."""
        if len(stracks) <= 0:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][6] = 0
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = BOTrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    def convert_coords(self, tlwh: np.ndarray) -> np.ndarray:
        """Convert tlwh bounding box coordinates to xywh format."""
        return self.tlwh_to_xywh(tlwh)

    @staticmethod
    def tlwh_to_xywh(tlwh: np.ndarray) -> np.ndarray:
        """Convert bounding box from tlwh (top-left-width-height) to xywh (center-x-center-y-width-height) format."""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret


class BOTSORT(BYTETracker):
    # Persistent ID API
    @staticmethod
    def add_persistent_track(id, tracker=None):
        PersistentIDRegistry.add_persistent_track(id, tracker=tracker)
    remove_persistent_track = PersistentIDRegistry.remove_persistent_track
    is_persistent = PersistentIDRegistry.is_persistent
    """An extended version of the BYTETracker class for YOLO, designed for object tracking with ReID and GMC algorithm.

    Attributes:
        proximity_thresh (float): Threshold for spatial proximity (IoU) between tracks and detections.
        appearance_thresh (float): Threshold for appearance similarity (ReID embeddings) between tracks and detections.
        encoder (Any): Object to handle ReID embeddings, set to None if ReID is not enabled.
        gmc (GMC): An instance of the GMC algorithm for data association.
        args (Any): Parsed command-line arguments containing tracking parameters.

    Methods:
        get_kalmanfilter: Return an instance of KalmanFilterXYWH for object tracking.
        init_track: Initialize track with detection results and optional image for ReID.
        get_dists: Get distances between tracks and detections using IoU and (optionally) ReID.
        multi_predict: Predict the mean and covariance of multiple object tracks using a shared Kalman filter.
        reset: Reset the BOTSORT tracker to its initial state.

    Examples:
        Initialize BOTSORT and process detections
        >>> bot_sort = BOTSORT(args, frame_rate=30)
        >>> bot_sort.init_track(results, img)
        >>> bot_sort.multi_predict(tracks)

    Notes:
        The class is designed to work with a YOLO object detection model and supports ReID only if enabled via args.
    """

    def __init__(self, args: Any, frame_rate: int = 30):
        """Initialize BOTSORT object with ReID module and GMC algorithm.

        Args:
            args (Any): Parsed command-line arguments containing tracking parameters.
            frame_rate (int): Frame rate of the video being processed.
        """
        super().__init__(args, frame_rate)
        self.gmc = GMC(method=args.gmc_method)

        # ReID module
        self.proximity_thresh = args.proximity_thresh
        self.appearance_thresh = args.appearance_thresh
        self.encoder = (
            (lambda feats, s: [f.cpu().numpy() for f in feats])  # native features do not require any model
            if args.with_reid and self.args.model == "auto"
            else ReID(args.model)
            if args.with_reid
            else None
        )

    def get_kalmanfilter(self) -> KalmanFilterXYWH:
        """Return an instance of KalmanFilterXYWH for predicting and updating object states in the tracking process."""
        return KalmanFilterXYWH()

    def init_track(self, results, img: np.ndarray | None = None) -> list[BOTrack]:
        """Initialize object tracks using detection bounding boxes, scores, class labels, and optional ReID features."""
        if len(results) == 0:
            return []
        bboxes = results.xywhr if hasattr(results, "xywhr") else results.xywh
        bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)
        if self.args.with_reid and self.encoder is not None:
            features_keep = self.encoder(img, bboxes)
            return [BOTrack(xywh, s, c, f) for (xywh, s, c, f) in zip(bboxes, results.conf, results.cls, features_keep)]
        else:
            return [BOTrack(xywh, s, c) for (xywh, s, c) in zip(bboxes, results.conf, results.cls)]

    def get_dists(self, tracks: list[BOTrack], detections: list[BOTrack]) -> np.ndarray:
        """Calculate distances between tracks and detections using IoU and optionally ReID embeddings."""
        dists = matching.iou_distance(tracks, detections)
        dists_mask = dists > (1 - self.proximity_thresh)

        if self.args.fuse_score:
            dists = matching.fuse_score(dists, detections)

        if self.args.with_reid and self.encoder is not None:
            emb_dists = matching.embedding_distance(tracks, detections) / 2.0
            emb_dists[emb_dists > (1 - self.appearance_thresh)] = 1.0
            emb_dists[dists_mask] = 1.0
            dists = np.minimum(dists, emb_dists)
        return dists

    def multi_predict(self, tracks: list[BOTrack]) -> None:
        """Predict the mean and covariance of multiple object tracks using a shared Kalman filter."""
        BOTrack.multi_predict(tracks)

    def update(self, results, img: np.ndarray | None = None, feats: np.ndarray | None = None) -> np.ndarray:
        """Update tracker with persistent ID support: snapshot embeddings for lost tracks and run Round 3 matching."""
        from scipy.spatial.distance import cdist

        # --- Step 1: Snapshot embeddings for persistent tracks currently in lost_stracks ---
        # We do this every frame they're lost so by the time max_time_lost is exceeded we always
        # have the freshest possible embedding — not a stale one from when they first went missing.
        for track in self.lost_stracks:
            tid = int(track.track_id)
            if self.is_persistent(tid):
                registry_cls = PersistentIDRegistry.persistent_info[tid]["cls"]
                LOGGER.info(f"[SNAPSHOT] id={tid} track.cls={track.cls} registry_cls={registry_cls} feat_available={track.smooth_feat is not None}")
                if track.smooth_feat is not None:
                    # Only snapshot embedding if class matches — prevents corrupting
                    # the embedding when BOT-SORT reassigns this ID to a different object
                    if registry_cls is None or int(track.cls) == int(registry_cls):
                        PersistentIDRegistry.persistent_info[tid]["embedding"] = track.smooth_feat.copy()
                    if registry_cls is None:
                        PersistentIDRegistry.persistent_info[tid]["cls"] = track.cls

        # --- Run normal BOT-SORT Rounds 1 and 2 ---
        output = super().update(results, img, feats)

        # --- Step 2: Round 3 — match unactivated detections against persistent registry ---
        # After super().update(), any detection that went unmatched was either turned into a new
        # track (activated) or discarded. We need to intercept the new tracks that were just
        # activated this frame and check if they should instead be re-assigned a persistent ID.
        # These are tracks with tracklet_len == 0 and is_activated == True added this frame.
        if not PersistentIDRegistry.persistent_info or not self.args.with_reid:
            return output

        # Collect candidates: registry entries that have a valid embedding
        persistent_ids = [
            pid for pid, info in PersistentIDRegistry.persistent_info.items()
            if info["embedding"] is not None
        ]
        if not persistent_ids:
            return output

        # Find newly activated tracks this frame (just born, tracklet_len == 0)
        # These are the detections that didn't match any existing track
        new_tracks = [t for t in self.tracked_stracks if t.is_activated and t.tracklet_len == 0]
        if not new_tracks:
            return output

        # Build registry embedding matrix: shape (num_persistent, feat_dim)
        registry_embeddings = np.array(
            [PersistentIDRegistry.persistent_info[pid]["embedding"] for pid in persistent_ids],
            dtype=np.float32,
        )
        registry_classes = [PersistentIDRegistry.persistent_info[pid]["cls"] for pid in persistent_ids]

        # Track which persistent IDs have already been claimed this frame
        # so that only one detection can be re-assigned per persistent ID
        claimed_pids = set()

        for new_track in new_tracks:
            if new_track.curr_feat is None:
                continue

            det_feat = new_track.curr_feat.reshape(1, -1).astype(np.float32)  # (1, feat_dim)

            # Filter candidates by class
            class_mask = [
                i for i, pid in enumerate(persistent_ids)
                if registry_classes[i] is not None and int(registry_classes[i]) == int(new_track.cls)
            ]
            if not class_mask:
                continue

            candidate_ids = [persistent_ids[i] for i in class_mask]
            candidate_embeddings = registry_embeddings[class_mask]  # (num_candidates, feat_dim)

            # Cosine distance: shape (1, num_candidates)
            dists = cdist(det_feat, candidate_embeddings, metric="cosine")[0]  # (num_candidates,)

            best_idx = int(np.argmin(dists))
            best_dist = dists[best_idx]
            best_pid = candidate_ids[best_idx]

            # Check against appearance threshold (convert similarity thresh to distance: 1 - thresh)
            if best_dist > (1 - self.appearance_thresh):
                continue

            # Skip if this persistent ID was already claimed by a better match this frame
            if best_pid in claimed_pids:
                continue

            claimed_pids.add(best_pid)

            LOGGER.info(f"[ROUND3] new_track id={new_track.track_id} cls={int(new_track.cls)} "
            f"best_pid={best_pid} registry_cls={PersistentIDRegistry.persistent_info[best_pid]['cls']} "
            f"dist={best_dist:.3f} thresh={1 - self.appearance_thresh:.3f} WILL_MATCH={best_dist <= (1 - self.appearance_thresh)}")

            # Re-assign this new track the persistent ID
            # We forcibly set track_id to the persistent ID — no new ID counter increment
            new_track.track_id = best_pid
            # Ensure the global counter never re-issues this persistent ID to a future new track
            if best_pid >= BaseTrack._count:
                BaseTrack._count = best_pid + 1
            # Update the registry embedding with the fresh detection feature
            PersistentIDRegistry.persistent_info[best_pid]["embedding"] = new_track.smooth_feat.copy()
            # Never overwrite cls — it was set at registration and must stay as ground truth
            if PersistentIDRegistry.persistent_info[best_pid]["cls"] is None:
                PersistentIDRegistry.persistent_info[best_pid]["cls"] = new_track.cls

        return output

    def reset(self) -> None:
        """Reset the BOTSORT tracker to its initial state, clearing all tracked objects and internal states."""
        super().reset()
        self.gmc.reset_params()

    def _should_remove_track(self, track_id):
        # Only remove if not persistent
        return not self.is_persistent(track_id)


class ReID:
    """YOLO model as encoder for re-identification."""

    def __init__(self, model: str):
        """Initialize encoder for re-identification.

        Args:
            model (str): Path to the YOLO model for re-identification.
        """
        from ultralytics import YOLO

        self.model = YOLO(model)
        self.model(embed=[len(self.model.model.model) - 2 if ".pt" in model else -1], verbose=False, save=False)  # init

    def __call__(self, img: np.ndarray, dets: np.ndarray) -> list[np.ndarray]:
        """Extract embeddings for detected objects."""
        feats = self.model.predictor(
            [save_one_box(det, img, save=False) for det in xywh2xyxy(torch.from_numpy(dets[:, :4]))]
        )
        if len(feats) != dets.shape[0] and feats[0].shape[0] == dets.shape[0]:
            feats = feats[0]  # batched prediction with non-PyTorch backend
        return [f.cpu().numpy() for f in feats]