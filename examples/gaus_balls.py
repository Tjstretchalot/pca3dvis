"""Generates some data from random gaussian blobs and renders it"""
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pca3dvis.pcs as pcs
import pca3dvis.worker as worker
import numpy as np

FEATURES = 10
"""The embedded space of the generated data. Every later snapshot
has one more feature just to see that doesn't hurt anything"""
CLUSTERS = 5
"""How many clusters are made in the embedding space"""
SNAPSHOTS = 2
"""How many "snapshots" we generate"""
SAMPLES_PER_CLUST = 200
"""How many samples in each cluster"""
CLUST_STD = 0.2
"""Standard deviation of each cluster"""
DRAFT = True
"""If draft settings are used for the video (ie. lower quality, but faster)"""

def gaus_ball(center: np.ndarray, std: int, num_samples: int):
    """Produces a gaussian ball with the given center and std deviation"""
    return (
        np.random.randn(num_samples, center.shape[0]).astype(center.dtype)
        * std + center
    )

def _main():
    # First generate some data!
    datas = []
    for snap in range(SNAPSHOTS):
        # We are looping over each of the snapshots we have. Since we are
        # generating points randomly, these snapshots are meaningless, but
        # we will see how these points are rendered

        centers = np.random.uniform(-2, 2, (CLUSTERS, FEATURES + snap))
        # Get a center for each cluster uniformly on a cube with sidelengths
        # 4 centered at the origin

        data = np.concatenate(
            tuple(
                gaus_ball(cent, CLUST_STD, SAMPLES_PER_CLUST)
                for cent in centers
            ),
            0
        )
        # Sample 200 points from each cluster and append them to the data
        # (but faster)

        datas.append(data)

    lbls = np.concatenate(
        tuple(
            np.zeros((SAMPLES_PER_CLUST,), data.dtype) + i
            for i in range(CLUSTERS)
        ),
        0
    )
    # Each cluster has its own label

    cmap = plt.get_cmap('Set1')
    markers = [ # An array with 1 marker for each point
        ( # The first marker
            np.ones(lbls.shape, dtype='bool'), # a mask containing every point
            {
                'c': lbls, # color these points based on their label
                'cmap': cmap, # to decide to color from the label, use the colormap
                's': 20, # the points should be around 20px
                'marker': 'o', # use a circle to represent these points
                'norm': mcolors.Normalize(0, CLUSTERS - 1) # the smallest label is 0, largest is CLUSTERS-1
            }
        )
    ]

    proj = pcs.get_pc_trajectory(datas, lbls)
    # This performs linear dimensionality-reduction using principal component
    # analysis to get the three-dimensional points we can actually plot

    worker.generate(
        proj,  # Plot the points we found
        markers,  # use the markers we made earlier
        ['Gaussian Balls (1)', 'Gaussian Balls (2)'], # title the different slices as so
        'out/examples/gaus_balls',  # store the result in this folder
        DRAFT # determines if we should store low quality (if True) or high quality (if False)
    )
    # That's it!

if __name__ == '__main__':
    _main()
