# PCA 3D Visualizations

This module produces informative visualizations of atleast 3-dimensional data,
optionally using principal component analysis and automatic clustering.

## Features

- *Animated*: high-quality videos with multi-processed video production using
[pympanim](https://github.com/Tjstretchalot/pympanim).
- (Pdf) *snapshots* that can be used in research papers or where the full video
cannot be used.
- Supports automatic *clustering* for informative zooms
- Supports *trajectories* of data, for example for visualizing data through
time for a recurrent network or through layers of a feedforward network.
- Supports using different markers, in any combination of style and color

## Installation

This package requires [ffmpeg](https://ffmpeg.org/) to be installed.

`pip install pca3dvis`

## Example Videos

`examples/gaus_balls.py`

https://youtu.be/n9rpWhuN6LA

`examples/all_rand.py`

https://www.youtube.com/watch?v=YnIRqSQ8lAU

## Usage

Video introduction at https://youtu.be/JqfVY9pdxY8

### Projecting to 3D from higher dimensions

There are many approaches for projecting to 3-dimensions. Once you have used
any of these projections, you can use this library to visualize the resulting
3d scatter plot.

This library provides one linear projection that depends on the eigenvectors
of the covariance matrix, called principal component analysis. For more
information see
[Wold, Esbensen, and Geladi, 1987](https://www.sciencedirect.com/science/article/pii/0169743987800849)

To quickly generate a `pca3dvis.trajectory.ProjectedTrajectory` using this
technique for an arbitrary `ndarray[samples, features]` where `features > 3`,
the following snippet will work:

```py
import numpy as np
import pca3dvis.pcs as pcs

data: np.ndarray # must have shape [samples > 3, features > 3]
lbls: np.ndarray # must have shape [samples = data.shape[0], ...]
traj = pcs.get_pc_trajectory([data], lbls)
```

This creates a trajectory with a single snapshot; for multiple snapshots, just
have `[data]` instead be a list of ndarrays which each have the same number of
samples but possibly different numbers of features.

### Converting 3D data to trajectories

A trajectory is a sequence of one or more snapshots which have the same labels.
They include the projection matrix that created them from the original data.
For quickly plotting already projected data, the labels can be swapped with all
zeros of the appropriate shape and the projection matrix with the identity map.

```py
import numpy as np
import pca3dvis.snapshot
import pca3dvis.trajectory

raw_data: np.ndarray # must have shape [samples > 3, og_features]
proj_matrices: np.ndarray # must have shape [og_features, 3]
lbls: np.ndarray # must have shape [samples = data.shape[0], ...]
snap = pca3dvis.snapshot.project_with_matrix(raw_data, lbls, proj_matrices)

traj = pca3dvis.trajectory.ProjectedTrajectory([snap]) # trajectory of one snap
```

### Plotting trajectories

Plotting trajectories, the main feature of this module, requires only that you
give a name to each snapshot and give markers to each label.

Markers are described as follows: Suppose there are N samples and M distinct
styles. Then for each style m, you will create a bool numpy array with shape
(N,) mask where `mask[i]` is `True` if `sample[i]` should be rendered with
style `m`. The style is described with a `dict` which are the keyword arguments
to [scatter](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.scatter.html#matplotlib.axes.Axes.scatter).
It should at least specifiy the size and color (`s` and `c` respectively), and
it may use a [colormap](https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html)
to do so, but if it does it must specify the [norm](https://matplotlib.org/3.1.1/tutorials/colors/colormapnorms.html)
exactly, i.e., with vmin and vmax not None.

For a more complete example of styling, see `examples/gaus_balls.py`.

```py
import numpy as np
import pca3dvis.trajectory
import pca3dvis.worker

traj: pca3dvis.trajectory.ProjectedTrajectory
titles = [f'Snapshot {i+1}' for i in range(traj.num_snapshots)]
markers = [(
    np.ones(traj.num_samples, 'bool'),
    {'s': 20, 'c': 'tab:red'}
)]
pca3dvis.worker.generate(traj, markers, titles, 'out/my_out_folder',
                         draft=True, clusters=True)
```
