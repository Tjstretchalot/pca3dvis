"""Describes a generic pca snapshot and methods for generating them from raw
data."""

import numpy as np
import pytypeutils as tus

class ProjectedSnapshot:
    """Describes a linear projection of samples onto a different dimensional
    space.

    Attributes:
        projection_vectors (np.ndarray[og_size, proj_size]):
            The matrix which projects samples into the projected snapshot
            space. If A is projection_vectors and v a raw sample, then
            Av is the projected sample.

        projected_samples (np.ndarray[samples, proj_size]):
            The samples in this snapshot, already projected into the
            snapshot space.

        projected_sample_labels (np.ndarray[samples, ...]):
            The labels associated with each sample. For a sample i, its
            location in projection space is projected_samples[i] and it has
            the label projected_sample_labels[i]. The label may have any
            shape and any dtype.
    """
    def __init__(self, projection_vectors: np.ndarray,
                 projected_samples: np.ndarray,
                 projected_sample_labels: np.ndarray):
        tus.check_ndarrays(
            projection_vectors=(
                projection_vectors,
                ('og_size', 'proj_size'),
                ('float32', 'float64')
            ),
            projected_samples=(
                projected_samples,
                ('samples', ('proj_size', projection_vectors.shape[1])),
                projection_vectors.dtype
            )
        )
        tus.check(projected_sample_labels=(
            projected_sample_labels, np.ndarray))
        if projected_sample_labels.shape[0] != projected_samples.shape[0]:
            raise ValueError(
                'projected_sample_labels should have shape (samples, ...)'
                + f'where samples={projected_samples.shape[0]}'
                + '=projected_samples.shape[0], but has shape '
                + str(projected_sample_labels.shape))
        self.projection_vectors = projection_vectors
        self.projected_samples = projected_samples
        self.projected_sample_labels = projected_sample_labels

    @property
    def dtype(self):
        """Returns the data type for the projected samples and the projection
        vectors"""
        return self.projection_vectors.dtype

    @property
    def original_size(self):
        """Returns the number of features in the original space"""
        return self.projection_vectors.shape[0]

    @property
    def projection_size(self):
        """Returns the number of features in the projected space"""
        return self.projection_vectors.shape[1]

    @property
    def num_samples(self):
        """Returns the number of samples in this snapshot"""
        return self.projected_samples.shape[0]

    def save(self, outfile: str):
        """Saves this snapshot to the given file. The file should have no
        extension or have the extension '.npz'.

        Args:
            outfile (str): where to save this snapshot to
        """
        tus.check(outfile=(outfile, str))
        np.savez_compressed(
            outfile,
            projection_vectors=self.projection_vectors,
            projected_samples=self.projected_samples,
            projected_sample_labels=self.projected_sample_labels)

    @classmethod
    def load(cls, infile: str):
        """Loads this snapshot from the given file. The file should be as
        passed to save.

        Args:
            infile (str): where to load this snapshot from
        """
        tus.check(infile=(infile, str))
        with np.load(infile) as npin:
            return cls(npin['projection_vectors'],
                       npin['projected_samples'],
                       npin['projected_sample_labels'])

def project_with_matrix(samples: np.ndarray, labels: np.ndarray,
                        mat: np.ndarray) -> ProjectedSnapshot:
    """Projects the given samples using the given projection
    matrix.

    Arguments:
        samples (np.ndarray[samples, features]): the samples to project
        labels (np.ndarray[samples, ...]): the labels for each sample
        mat (np.ndarray[features, proj_size]):

    Returns:
        snap (ProjectedSnapshot): the samples projected using the matrix
    """
    tus.check_ndarrays(
        samples=(samples, ('samples', 'features'), ('float32', 'float64')),
        mat=(
            mat,
            (('features', samples.shape[1]), 'proj_size'),
            samples.dtype
        ))

    return ProjectedSnapshot(mat, samples @ mat, labels)
