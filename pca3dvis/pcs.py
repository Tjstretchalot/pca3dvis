"""Calculates principal components and principal component vectors"""

import typing
import numpy as np
import scipy.linalg
import pytypeutils as tus
import pca3dvis.matching as matching
from pca3dvis.snapshot import project_with_matrix
from pca3dvis.trajectory import ProjectedTrajectory

def get_pcs(mat: np.ndarray, num_pcs: int) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Gets the top num_pcs principal component vectors and associated
    values from the specified matrix.

    Args:
        mat (np.ndarray[samples, features]): The matrix to reduce
        num_pcs (int): the number of pcs to calculate

    Returns:
        eigs (np.ndarray[num_pcs]): the relative importance in descending order
            for each principal component vector
        eig_vecs (np.ndarray[features, num_pcs]): the projection matrix
    """
    tus.check_ndarrays(
        mat=(mat, ('samples', 'features'), ('float32', 'float64'))
    )
    tus.check(num_pcs=(num_pcs, int))

    mean_cent = mat - mat.mean(0)
    cov = np.cov(mean_cent.T)
    eig, eig_vecs = scipy.linalg.eig(cov)
    eig = np.real(eig)
    ind = np.argsort(np.abs(eig))[::-1]
    eig_vecs = np.real(eig_vecs[:, ind])
    eig = eig[ind]

    eig_vecs = eig_vecs[:, :num_pcs]
    eig = eig[:num_pcs]

    return eig, eig_vecs

def get_pc_trajectory(mats: typing.Tuple[np.ndarray], lbls: np.ndarray,
                      num_pcs: int = 3, match: bool = True) -> ProjectedTrajectory:
    """Gets the matched pc trajectory for the given raw sample
    points. Note that the axis do not correspond to the principal component
    vectors because we allow orthogonal transformations. This can be
    disabled with match=False

    Args:
        mat (tuple[np.ndarray]): Corresponds to the non-projected data at each
            sample. Each array should be [samples, layer_features] where layer
            features may be different for each array but samples should be the
            same
        lbls (np.ndarray[samples, ...]): the labels corresponding to each
            sample, which may have any dtype and any shape so long as the
            first dimension is which sample
        num_pcs (int): the number of pcs in the projected trajectory.
            Default 3
        match (bool): if True we match each snapshot so that they are as
            similar as possible while preserving the geometry (shape and
            magnitude). In other words, we rotate later snapshots to get
            them to appear most similar to the one before it. Default True
    Returns:
        ProjectedTrajectory: The pc-trajectory for the given samples
    """
    tus.check(
        mats=(mats, (list, tuple)),
        lbls=(lbls, np.ndarray),
        num_pcs=(num_pcs, int)
    )
    tus.check_listlike(mats=(mats, np.ndarray, (1, None)))
    tus.check_ndarrays(
        **{'mats[0]': (
            mats[0], ('samples', 'layer_features'), ('float32', 'float64'))
          })
    samples = mats[0].shape[0]
    for i, arr in enumerate(mats):
        tus.check_ndarrays(
            **{f'mats[{i}]': (
                arr,
                (('samples', samples), 'layer_features'),
                mats[0].dtype
            )}
        )

    if lbls.shape[0] != samples:
        raise ValueError(f'expected lbls.shape = (samples, ...) but got '
                         + f'{lbls.shape} (samples={samples})')

    if num_pcs < 1:
        raise ValueError(f'cannot find {num_pcs} pcs (must be positive)')

    snapshots = []
    for i, mat in enumerate(mats):
        _, eig_vecs = get_pcs(mat, num_pcs)

        snap = project_with_matrix(mat, lbls, eig_vecs)
        if match and i > 0:
            snap = matching.match_snaps(snapshots[i - 1], snap)
        snapshots.append(snap)

    return ProjectedTrajectory(snapshots)
