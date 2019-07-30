"""Generates some data from random gaussian blobs and renders it"""
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pca3dvis.pcs as pcs
import pca3dvis.worker as worker
import numpy as np

FEATURES = 10
"""The embedded space of the generated data. Every later snapshot
has one more feature just to verify that doesn't hurt anything"""
CLUSTERS = 5
"""How many clusters are made in the embedding space"""
SNAPSHOTS = 2
"""How many "snapshots" we generate"""
SAMPLES_PER_CLUST = 200
"""How many samples in each cluster"""
CLUST_STD = 0.2
"""Standard deviation of each cluster"""
DRAFT = True
"""If draft settings are used"""

def gaus_ball(center: np.ndarray, std: int, num_samples: int):
    """Produces a gaussian ball with the given center and std deviation"""
    return (
        np.random.randn(num_samples, center.shape[0]).astype(center.dtype)
        * std + center
    )

def _main():

    datas = []
    for snap in range(SNAPSHOTS):
        centers = np.random.uniform(-2, 2, (CLUSTERS, FEATURES + snap))
        data = np.concatenate(
            tuple(
                gaus_ball(cent, CLUST_STD, SAMPLES_PER_CLUST)
                for cent in centers
            ),
            0
        )
        datas.append(data)

    lbls = np.concatenate(
        tuple(
            np.zeros((SAMPLES_PER_CLUST,), data.dtype) + i
            for i in range(CLUSTERS)
        ),
        0
    )

    cmap = plt.get_cmap('Set1')
    markers = [
        (
            np.ones(lbls.shape, dtype='bool'),
            {
                'c': lbls,
                'cmap': cmap,
                's': 20,
                'marker': 'o',
                'norm': mcolors.Normalize(0, CLUSTERS - 1)
            }
        )
    ]

    proj = pcs.get_pc_trajectory(datas, lbls)

    worker.generate(
        proj, markers, ['Gaussian Balls (1)', 'Gaussian Balls (2)'],
        'out/examples/gaus_balls', DRAFT
    )

if __name__ == '__main__':
    _main()
