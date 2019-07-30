"""Contains various scenes which are generally useful for producing projected
trajectory visualizations. These scenes all work with state.ProjectedState.
"""
import typing
import pytypeutils as tus
import numpy as np
import pympanim.acts as acts
import pca3dvis.state as state
from pca3dvis.snapshot import ProjectedSnapshot

def _linear_to(start, end, perc):
    return start + (end - start) * perc

class FixedTitleScene(acts.Scene):
    """When entered, sets the title to a specified value.

    Attributes:
        title (str): the title to set
    """
    def __init__(self, title: str):
        self.title = title

    def enter(self, act_state: state.ProjectedState):
        act_state.title = self.title

class FixedZoomScene(acts.Scene):
    """When entered, sets the zoom to a specific value.

    Attributes:
        zoom (np.ndarray): the zoom to fix to
    """
    def __init__(self, zoom: np.ndarray):
        self.zoom = zoom

    def enter(self, act_state: state.ProjectedState):
        act_state.zoom = self.zoom

class FixedRotationScene(acts.Scene):
    """When entered, sets the rotation to a specific value

    Attributes:
        rotation (tuple[float, float]): the rotation (elevation and
            azimuth angle)
    """
    def __init__(self, rotation: typing.Tuple[float, float]):
        self.rotation = rotation

    def enter(self, act_state: state.ProjectedState):
        act_state.rotation = self.rotation

class SnapshotScene(acts.Scene):
    """When entered sets the visible points to correspond to a particular
    snapshot.

    Attributes:
        snapshot_ind (int): the index in the trajectory
    """
    def __init__(self, snapshot_ind: int):
        self.snapshot_ind = snapshot_ind

    def enter(self, act_state: state.ProjectedState):
        act_state.set_snapshot_visible(self.snapshot_ind)

class MaskedScene(acts.Scene):
    """When entered sets the visible points to correspond specifically to a
    mask of a particular snapshot.

    Attributes:
        snapshot_ind (int): the index in the trajectory
        mask (np.ndarray): a mask on the points
    """
    def __init__(self, snapshot_ind: int, mask: np.ndarray):
        tus.check_ndarrays(mask=(mask, ('samples',), 'bool'))
        self.snapshot_ind = snapshot_ind
        self.mask = mask

    def enter(self, act_state: state.ProjectedState):
        masked_pts = (
            act_state.trajectory.snapshots[self.snapshot_ind].projected_samples
        )[self.mask]
        act_state.visible_points = ((masked_pts, self.mask, dict()),)

class FadeScene(acts.Scene):
    """Takes all the masked samples fades them out, leaving the other samples
    to have the default values. Does not effect rotation or zoom and works in
    unit time.

    Attributes:
        snapshot_ind (int): which snapshot we are fading
        mask (np.ndarray): the mask for which points should be faded out

        alpha_start (float): the alpha value to start with
        alpha_end (float): the alpha value to end with
    """
    def __init__(self, snapshot_ind: int, mask: np.ndarray, alpha_start: float,
                 alpha_end: float):
        tus.check_ndarrays(mask=(mask, ('samples',), 'bool'))
        self.snapshot_ind = snapshot_ind
        self.mask = mask
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end

    def apply(self, act_state: state.ProjectedState, time_ms: float,
              dbg: bool = False):
        pts = (act_state.trajectory.snapshots[self.snapshot_ind]
               .projected_samples)
        imask = ~self.mask
        alpha = _linear_to(self.alpha_start, self.alpha_end, time_ms)
        act_state.visible_points = (
            (pts[self.mask], self.mask, {'alpha': alpha}),
            (pts[imask], imask, dict())
        )

class RotationScene(acts.Scene):
    """Changes the rotation linearly between the given amounts in unit time

    Attributes:
        rotation_start (tuple[float, float]): the rotation to start at
        rotation_end (tuple[float, float]): the rotation to end at
    """
    def __init__(self, rotation_start: typing.Tuple[float, float],
                 rotation_end: typing.Tuple[float, float]):
        self.rotation_start = rotation_start
        self.rotation_end = rotation_end

    def apply(self, act_state: state.ProjectedState, time_ms: float,
              dbg: bool = False):
        act_state.rotation = (
            _linear_to(
                self.rotation_start[0], self.rotation_end[0], time_ms) % 360,
            _linear_to(
                self.rotation_start[1], self.rotation_end[1], time_ms) % 360
        )

class ZoomScene(acts.Scene):
    """Changes the x/y/z limits linearly between the given amounts in unit time

    Attributes:
        zoom_start (np.ndarray[3, 2]): the initial zoom
        zoom_end (np.ndarray[3, 2]): the final zoom
    """
    def __init__(self, zoom_start: np.ndarray, zoom_end: np.ndarray):
        self.zoom_start = zoom_start
        self.zoom_end = zoom_end

    def apply(self, act_state: state.ProjectedState, time_ms: float,
              dbg: bool = False):
        act_state.zoom = _linear_to(self.zoom_start, self.zoom_end, time_ms)

class InterpScene(acts.Scene):
    """Moves the points between two snapsots linearly in unit time

    Attributes:
        start_snapshot (int): the snapshot to start at
        end_snapshot (int): the snapshot to end at
    """
    def __init__(self, start_snapshot: int, end_snapshot: int):
        self.start_snapshot = start_snapshot
        self.end_snapshot = end_snapshot

    def apply(self, act_state: state.ProjectedState, time_ms: float,
              dbg: bool = False):
        snap1: ProjectedSnapshot = (
            act_state.trajectory.snapshots[self.start_snapshot])
        snap2: ProjectedSnapshot = (
            act_state.trajectory.snapshots[self.end_snapshot])

        act_state.visible_points = ((
            _linear_to(snap1.projected_samples,
                       snap2.projected_samples,
                       time_ms),
            np.ones(snap1.num_samples, dtype='bool'),
            dict()),)
