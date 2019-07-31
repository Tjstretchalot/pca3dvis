"""Ultra-simple example that just displays some random 3D data."""

import numpy as np
import pca3dvis.snapshot
import pca3dvis.trajectory
import pca3dvis.worker

def _main():
    arr = np.random.uniform(-1, 1, (1000, 3))
    proj = np.eye(3, 3)
    lbl = np.zeros(arr.shape[0], dtype='int32')

    snap = pca3dvis.snapshot.project_with_matrix(arr, lbl, proj)
    traj = pca3dvis.trajectory.ProjectedTrajectory([snap])

    markers = [
        (
            np.ones(lbl.shape, 'bool'),
            {'s': 20, 'c': 'tab:red'}
        )
    ]

    pca3dvis.worker.generate(traj, markers, ['Random'], 'out/examples/allrand')

if __name__ == '__main__':
    _main()
