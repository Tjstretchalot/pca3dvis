"""Describes a trajectory of projected samples, which is a series
of projected snapshots which use the same underlying labels.
"""

import typing
import pytypeutils as tus
import pca3dvis.snapshot as snapshot

class ProjectedTrajectory:
    """Describes a path of points which has the same number of samples at
    each step and uses the same labels at each step. The underlying points
    and/or the projection matrices may change at each step.

    Attributes:
        snapshots (tuple[pcs.ProjectedSnapshot]): the snapshots that make up
            this trajectory
    """
    def __init__(self, snapshots: typing.List[snapshot.ProjectedSnapshot]):
        tus.check(snapshots=(snapshots, (list, tuple)))
        tus.check_listlike(
            snapshots=(snapshots, snapshot.ProjectedSnapshot, (1, None)))

        num_samples = snapshots[0].num_samples
        proj_size = snapshots[0].projection_size
        dtype = snapshots[0].dtype
        lab_shape = snapshots[0].projected_sample_labels.shape
        lab_dtype = snapshots[0].projected_sample_labels.dtype

        # For the sake of performance when there are many samples we only
        # verify the shape and datatype of the labels match, however they
        # should actually be allclose.

        for i, snap in enumerate(snapshots):
            if snap.num_samples != num_samples:
                raise ValueError(
                    f'snapshots[0].num_samples = {num_samples}, but '
                    + f'snapshots[{i}].num_samples = {snap.num_samples}'
                )
            if snap.projection_size != proj_size:
                raise ValueError(
                    f'snapshots[0].projection_size = {proj_size}, but '
                    + f'snapshots[{i}].projection_size = '
                    + str(snap.projection_size)
                )
            if snap.dtype != dtype:
                raise ValueError(
                    f'snapshots[0].dtype = {dtype}, but snapshots[{i}].dtype ='
                    + f' {snap.dtype}'
                )
            if snap.projected_sample_labels.shape != lab_shape:
                raise ValueError(
                    f'snapshots[0].projected_sample_labels.shape = {lab_shape}'
                    + f', but snapshots[{i}].projected_sample_labels.shape = '
                    + str(snap.projected_sample_labels.shape)
                )
            if snap.projected_sample_labels.dtype != lab_dtype:
                raise ValueError(
                    f'snapshots[0].projected_sample_labels.dtype = {lab_dtype}'
                    + f', but snapshots[{i}].projected_sample_labels.dtype = '
                    + str(snap.projected_sample_labels.dtype)
                )

        self.snapshots = tuple(snapshots)

    @property
    def num_snapshots(self):
        """Returns the number of snapshots in this trajectory"""
        return len(self.snapshots)

    @property
    def num_samples(self):
        """Returns the number of samples in each snapshot"""
        return self.snapshots[0].num_samples

    @property
    def projection_size(self):
        """Returns the proejction size of each snapshot"""
        return self.snapshots[0].projection_size
