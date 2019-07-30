"""Rhis module is meant for detecting clusters that occur within a projected
space. This uses one specific type of clustering that seems to work well for
3-dimensional projected data.
"""
import typing
import numpy as np

import sys
import os
try:
    stdout, stderr = sys.stdout, sys.stderr
    with open(os.devnull, 'w') as dnull:
        sys.stdout = dnull
        sys.stderr = dnull
        import hdbscan # emits a warning that is very difficult to suppress
except:
    sys.stdout = stdout
    sys.stderr = stderr
    raise
finally:
    sys.stdout = stdout
    sys.stderr = stderr

import pytypeutils as tus

class Clusters:
    """The data class which stores information about clusters generated from
    particular samples.

    Attributes:
        samples (ndarray[n_samples, n_features]): the samples that the clusters were
            selected from.
        centers (ndarray[n_clusters, n_features]): where the cluster centers are located
        labels (ndarray[n_samples]): each value is 0,1,...,n_clusters-1 and corresponds
            to the nearest cluster to the corresponding sample in pc-space

        calculate_params (dict[str, any]): the parameters that were used to generate these
            clusters.
    """

    def __init__(self, samples: np.ndarray, centers: np.ndarray, labels: np.ndarray,
                 calculate_params: typing.Dict[str, typing.Any]):
        tus.check(samples=(samples, np.ndarray), centers=(centers, np.ndarray),
                  labels=(labels, np.ndarray), calculate_params=(calculate_params, dict))
        tus.check_ndarrays(
            samples=(samples, ('n_samples', 'n_features'),
                     (np.dtype('float32'), np.dtype('float64'))),
            centers=(
                centers,
                ('n_clusters',
                 ('n_features', samples.shape[1] if len(samples.shape) > 1 else None)
                ),
                samples.dtype
            ),
            labels=(
                labels,
                (('n_samples', samples.shape[0] if bool(samples.shape) else None),),
                (np.dtype('int32'), np.dtype('int64'))
            )
        )
        self.samples = samples
        self.centers = centers
        self.labels = labels
        self.calculate_params = calculate_params

    @property
    def num_samples(self):
        """Returns the number of samples used to generate these clusters"""
        return self.samples.shape[0]

    @property
    def num_features(self):
        """Returns the number of features in the sample space"""
        return self.samples.shape[1]

    @property
    def num_clusters(self):
        """Returns the number of clusters found. This may have been chosen"""
        return self.centers.shape[0]

def find_clusters(samples: np.ndarray) -> Clusters:
    """Attempts to locate clusters in the given samples in the most generic
    way possible."""
    args = {
        'min_cluster_size': int(0.2*samples.shape[0]),
        'min_samples': 10
    }

    args_meta = {
        'method': 'hdbscan.HDBSCAN'
    }


    clusts = hdbscan.HDBSCAN(**args)
    clusts.fit(samples)

    # first we determine how many clusters there are which actually
    # have points belonging to them -1 is for unclustered points
    labels = clusts.labels_
    unique_labels = np.unique(labels)
    if -1 in unique_labels:
        unique_labels = np.ascontiguousarray(unique_labels[unique_labels != -1])

    # we are also going to want to centers of our labels
    sums = np.zeros((unique_labels.shape[0], samples.shape[1]), dtype='float64')
    num_per = np.zeros(unique_labels.shape[0], dtype='int64')
    new_labels = np.zeros(samples.shape[0], dtype='int32')

    # crunch numbers
    for lbl_ind, lbl in enumerate(unique_labels):
        mask = labels == lbl
        new_labels[mask] = lbl
        masked = samples[mask]
        sums[lbl_ind] = masked.sum(axis=0)
        num_per[lbl_ind] = masked.shape[0]

    if unique_labels.shape[0] == 1 and num_per[0] == samples.shape[0]:
        return Clusters(
            samples,
            np.zeros((0, samples.shape[1]), dtype='float32'),
            np.zeros((samples.shape[0],), dtype='int32'),
            {'clustering': args, 'other': args_meta}
        )

    # calculate centers of each cluster
    centers = (
        sums / (
            num_per.astype('float64')
            .reshape(num_per.shape[0], 1)
            .repeat(sums.shape[1], 1)
        )
    ).astype(samples.dtype)

    return Clusters(samples, centers, new_labels, {
        'clustering': args,
        'other': args_meta
    })
