# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track
from KCFpy import kcftracker

class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=30, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        #TODO: i don't need kf to control each track
        #self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1

    def predict(self, image):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            #track.predict(self.kf)
            track.predict(image)

    def update(self, detections, image):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            #self.tracks[track_idx].update(
            #    self.kf, detections[detection_idx])
            self.tracks[track_idx].update(detections[detection_idx], image)
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx], image)
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # cnn based appearance feature is not used in this kcf part
        # Update distance metric.
        # active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        # features, targets = [], []
        # for track in self.tracks:
        #     if not track.is_confirmed():
        #         continue
        #     features += track.features
        #     targets += [track.track_id for _ in track.features]
        #     track.features = []
        # self.metric.partial_fit(
        #     np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):


        def gated_metric(tracks, dets, track_indices, detection_indices):
            #print("gated metric")
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            #print("max and min in the cost matrix before gating: %f, %f" % (np.max(cost_matrix), np.min(cost_matrix)))
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix


        # just use feature similarity to see the importance of gated metric
        def just_feat_metric(tracks, dets, track_indices, detection_indices):
            #print("just feat metric")
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            #print(cost_matrix)
            #print("max and min in the cost matrix before gating: %f, %f" % (np.max(cost_matrix), np.min(cost_matrix)))
            #cost_matrix = linear_assignment.gate_cost_matrix(
            #    self.kf, cost_matrix, tracks, dets, track_indices,
            #    detection_indices)

            return cost_matrix

        # use iou cost and gated cost for cascade operation
        def gated_iou_metric(tracks, dets, track_indices, detection_indices):
            #print("gated iou metric")
            #features = np.array([dets[i].feature for i in detection_indices])
            #targets = np.array([tracks[i].track_id for i in track_indices])
            #cost_matrix = self.metric.distance(features, targets)
            cost_matrix = iou_matching.iou_cost(tracks, dets, track_indices, detection_indices)
            #print("max and min in the cost matrix before gating: %f, %f" % (np.max(cost_matrix), np.min(cost_matrix)))
            # TODO: remove gated cost provided by kalman fiter
            # cost_matrix = linear_assignment.gate_cost_matrix(
            #     self.kf, cost_matrix, tracks, dets, track_indices,
            #     detection_indices)


            return cost_matrix

        # just use iou similarity to see the importance of gated metric
        def just_iou_metric(tracks, dets, track_indices, detection_indices):
            #print("just iou metric")
            cost_matrix = iou_matching.iou_cost(tracks, dets, track_indices, detection_indices)
            #print("max and min in the cost matrix before gating: %f, %f" % (np.max(cost_matrix), np.min(cost_matrix)))
            #cost_matrix = linear_assignment.gate_cost_matrix(
            #    self.kf, cost_matrix, tracks, dets, track_indices,
            #    detection_indices)

            return cost_matrix


        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]


        """
        # Associate confirmed tracks using appearance features and Mahalanobis distance
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)
        """

        """
        # Associate confirmed tracks just using appearance
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                just_feat_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)
        """


        # Associate confirmed tracks using iou and Mahalanobis distance
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_iou_metric, self.max_iou_distance, self.max_age,
                self.tracks, detections, confirmed_tracks)


        """
        # Associate confirmed tracks just using iou
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                just_iou_metric, self.max_iou_distance, self.max_age,
                self.tracks, detections, confirmed_tracks)
        """


        #print("first phase track: ", matches_a)
        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        #print("second phase track: ", matches_b)
        matches = matches_a + matches_b
        #if len(matches) != 0:
        #    print("first stage: %.2f" % (len(matches_a) * 1.0 / len(matches) * 100))
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections


        #return matches_a, unmatched_tracks_a, unmatched_detections

    # def _initiate_track(self, detection):
    #     mean, covariance = self.kf.initiate(detection.to_xyah())
    #     self.tracks.append(Track(
    #         mean, covariance, self._next_id, self.n_init, self.max_age,
    #         detection.feature))
    #     self._next_id += 1

    def _initiate_track(self, detection, image):
        x, y, w, h = detection.tlwh
        kcf_tracker = kcftracker.KCFTracker(self._next_id, self.n_init, self.max_age,
                                            hog=True, fixed_window=True, multiscale=True)
        kcf_tracker.init([x, y, w, h], image)
        self.tracks.append(kcf_tracker)
        self._next_id += 1
