"""This module attempts to get two pieces of data to be as similar as possible
using only orthogonal transformations. Many analyses are unique only up to
orthonormal transformations

This module is specifically referring to matrices of shape
(in_features, out_features).

Restated, the goal is to minimize the frobenius distance between two matrices
using only an orthonormal matrix.

Restated, let A and B be matrices. First find

 R = argmin | NA - B |F
       N
    where N is orthonormal

And the result is (NA, B)

This problem is known as the Orthogonal Procrustes Problem and has a simple
solution using SVD: https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

It's straightforward to convert to minimizing (A - BN) by noting that forbenius
norms are invariant to transposes and multiplications by units (+/- 1)
"""

import numpy as np
import scipy.linalg
import pytypeutils as tus
import pca3dvis.snapshot as snapshot

def match(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Finds the matrix R that minimizes the frobenius norm of RA - B, where
    R is orthonormal.

    Args:
        a (np.ndarray[samples, features]): the first matrix to match
        b (np.ndarray[samples, features]): the second matrix to match

    Returns:
        np.ndarray: the orthonormal matching matrix R
    """
    tus.check_ndarrays(
        a=(a, ('samples', 'features'), ('float32', 'float64')),
        b=(b, (('samples', a.shape[0]), ('features', a.shape[1])), a.dtype)
    )
    m = b @ a.T
    u, _, vh = scipy.linalg.svd(m)
    return np.real(u @ vh)

def match_snaps(
        reference: snapshot.ProjectedSnapshot,
        tomatch: snapshot.ProjectedSnapshot) -> snapshot.ProjectedSnapshot:
    """Returns a new snapshot that is a transformation the tomatch snapshot
    to match the reference snapshot. Neither argument is modified.

    Reason why this works:

    Let A be reference projected samples.
    Let B be tomatch projected samples.

    Then |A - BN| = |(A-BN)'| = |A' - N'B'| = |N'B' - A'|
    """

    transform = match(tomatch.projected_samples.T, reference.projected_samples.T)
    return snapshot.ProjectedSnapshot(
        tomatch.projection_vectors @ transform.T,
        tomatch.projected_samples @ transform.T,
        tomatch.projected_sample_labels
    )
