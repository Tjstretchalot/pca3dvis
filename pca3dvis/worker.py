"""This provides the main glue code for using the module. It ties everything
together.
"""
import typing
import matplotlib.pyplot as plt
import numpy as np
import os
import pytypeutils as tus
import pympanim.acts as acts
import pympanim.easing
import pympanim.worker
import pytweening
import pca3dvis.trajectory
import pca3dvis.renderer
import pca3dvis.state
import pca3dvis.scenes as scenes
import pca3dvis.clusters

def generate(traj: pca3dvis.trajectory.ProjectedTrajectory,
             markers: typing.Tuple[typing.Tuple[np.ndarray, dict]],
             titles: typing.Tuple[str],
             outfolder: str,
             draft: bool = False,
             clusters: bool = True):
    """Generates a video and corresponding snapshots into the specified
    directory. If draft is true, this uses the draft settings. Otherwise,
    this uses high-quality production settings.

    For fine control over the settings it is recommended that you use the
    underlying pympanim functions.

    Args:
        traj (ProjectedTrajectory): the trajectory to plot
        markers (tuple[tuple[ndarray, dict]]):
            each
        titles (tuple[str]): one title per snapshot in the trajectory
        outfolder (str): where to save. must not already exist
        draft (bool): if true, lower quality settings are used
        clusters (bool): if clusters are detected and zoomed to
    """
    tus.check(
        traj=(traj, pca3dvis.trajectory.ProjectedTrajectory),
        markers=(markers, (list, tuple)),
        titles=(titles, (list, tuple)),
        outfolder=(outfolder, str),
        draft=(draft, bool),
        clusters=(clusters, bool)
    )
    for i, marker in enumerate(markers):
        tus.check(**{f'marker[{i}]': (marker, (list, tuple))})
        if len(marker) != 2:
            raise ValueError(
                f'expected marker[{i}] is (ndarray, dict), got {marker}')
        mask, kwargs = marker
        tus.check_ndarrays(**{
            f'marker[{i}][0]': (mask, (('samples', traj.num_samples),), 'bool')
        })
        tus.check(**{
            f'marker[{i}][1]': (kwargs, dict)
        })
    tus.check_listlike(titles=(titles, str, traj.num_snapshots))

    os.makedirs(outfolder)
    os.makedirs(os.path.join(outfolder, 'snapshots'))

    state = pca3dvis.state.ProjectedState(traj, markers)
    rend = pca3dvis.renderer.ProjectedRenderer(
        (19.2, 10.8) if not draft else (6.4, 4.8),
        100
    )

    for i, title in enumerate(titles):
        filepath = os.path.join(outfolder, 'snapshots', f'snapshot_{i}')
        ext = '.png' if draft else '.pdf'
        transp = not draft

        state.set_snapshot_visible(i, True)
        state.title = title
        for rot in range(15, 375, 60 if draft else 30):
            state.rotation = (30, rot)
            fig = rend.render_mpl(state)

            fname = filepath + f'_rot{rot}' + ext
            fig.savefig(fname, transparent=transp, dpi=rend.dpi)
            plt.close(fig)

    pts = traj.snapshots[0].projected_samples
    zoom = pca3dvis.state.get_square_bounds_for(pts)
    my_scene = (
        acts.FluentScene(scenes.SnapshotScene(0))
        .join(scenes.FixedTitleScene(titles[0]), False)
        .join(scenes.RotationScene((30, 45), (30, 45 + 360)), False)
        .join(scenes.FixedZoomScene(zoom), False)
        .dilate(pytweening.easeInOutSine)
        .time_rescale_exact(12, 's')
    )

    if clusters:
        _cluster_scene(my_scene, traj, 0, titles[0], draft)

    for snap_ind in range(1, traj.num_snapshots):
        snap = traj.snapshots[snap_ind]
        npts = snap.projected_samples
        nzoom = pca3dvis.state.get_square_bounds_for(npts)
        mzoom = pca3dvis.state.get_square_bounds_for_all((pts, npts))
        ntitle = titles[snap_ind]
        ititle = titles[snap_ind - 1] + ' -> ' + ntitle

        if not np.allclose(zoom, mzoom):
            (my_scene.push(scenes.ZoomScene(zoom, mzoom))
             .join(scenes.SnapshotScene(snap_ind - 1), False)
             .join(scenes.FixedTitleScene(ititle), False)
             .join(scenes.FixedRotationScene((30, 45)), False)
             .dilate(pympanim.easing.smoothstep)
             .time_rescale_exact(2, 's')
             .pop()
            )

        (my_scene.push(scenes.InterpScene(snap_ind - 1, snap_ind))
         .dilate(pytweening.easeInOutCirc)
         .join(scenes.FixedZoomScene(mzoom), False)
         .join(scenes.FixedTitleScene(ititle), False)
         .push(scenes.RotationScene((30, 45), (30, 45 + 360)))
         .dilate(pytweening.easeInOutSine)
         .dilate(pympanim.easing.squeeze, {'amt': 0.1})
         .pop('join')
         .time_rescale_exact(6, 's')
         .pop()
        )

        if not np.allclose(mzoom, nzoom):
            (my_scene.push(scenes.ZoomScene(mzoom, nzoom))
             .join(scenes.SnapshotScene(snap_ind), False)
             .join(scenes.FixedTitleScene(ititle), False)
             .join(scenes.FixedRotationScene((30, 45)), False)
             .dilate(pympanim.easing.smoothstep)
             .time_rescale_exact(2, 's')
             .pop()
            )

        (my_scene.push(scenes.SnapshotScene(snap_ind))
         .join(scenes.FixedTitleScene(ntitle), False)
         .join(scenes.RotationScene((30, 45), (30, 45 + 360)), False)
         .join(scenes.FixedZoomScene(nzoom), False)
         .dilate(pytweening.easeInOutSine)
         .time_rescale_exact(10, 's')
         .pop()
        )

        if clusters:
            _cluster_scene(my_scene, traj, snap_ind, ntitle, draft)

        pts = npts
        zoom = nzoom

    if draft:
        my_scene.time_rescale(5)

    pympanim.worker.produce(
        acts.Act(state, rend, [my_scene.build()]),
        60 if not draft else 30,
        100,
        -1,
        os.path.join(outfolder, 'video.mp4' if not draft else 'draft.mp4')
    )

def _cluster_scene(my_scene, traj, snap_ind, title, draft):
    snap = traj.snapshots[snap_ind]
    samps = snap.projected_samples
    clusts: pca3dvis.clusters.Clusters = pca3dvis.clusters.find_clusters(samps)
    outer_zoom = pca3dvis.state.get_square_bounds_for(samps)
    for clust in range(clusts.num_clusters):
        ctitle = f'{title} - Cluster {clust+1}'
        mask = clusts.labels == clust
        masked = samps[mask]
        inner_zoom = pca3dvis.state.get_square_bounds_for(masked)
        (my_scene
         .push(scenes.FadeScene(snap_ind, ~mask, 1, 0)) # FADE OUT UNMASKED
         .join(scenes.FixedTitleScene(ctitle), False)
         .join(scenes.FixedRotationScene((30, 45)), False)
         .join(scenes.FixedZoomScene(outer_zoom), False)
         .dilate(pytweening.easeOutSine)
         .time_rescale_exact(2, 's')
         .pop()
         .push(scenes.ZoomScene(outer_zoom, inner_zoom)) # ZOOM IN
         .join(scenes.MaskedScene(snap_ind, mask), False)
         .join(scenes.FixedTitleScene(ctitle), False)
         .join(scenes.FixedRotationScene((30, 45)), False)
         .dilate(pympanim.easing.smoothstep)
         .time_rescale_exact(2, 's')
         .pop()
         .push(scenes.RotationScene((30, 45), (30, 45 + 360))) # ROTATE
         .join(scenes.MaskedScene(snap_ind, mask), False)
         .join(scenes.FixedTitleScene(ctitle), False)
         .join(scenes.FixedZoomScene(inner_zoom), False)
         .dilate(pytweening.easeInOutSine)
         .time_rescale_exact(10, 's')
         .pop()
         .push(scenes.ZoomScene(inner_zoom, outer_zoom)) # ZOOM OUT
         .join(scenes.MaskedScene(snap_ind, mask), False)
         .join(scenes.FixedTitleScene(ctitle), False)
         .join(scenes.FixedRotationScene((30, 45)), False)
         .dilate(pympanim.easing.smoothstep)
         .time_rescale_exact(2, 's')
         .pop()
         .push(scenes.FadeScene(snap_ind, ~mask, 0, 1)) # FADE IN UNMASKED
         .join(scenes.FixedTitleScene(ctitle), False)
         .join(scenes.FixedRotationScene((30, 45)), False)
         .join(scenes.FixedZoomScene(outer_zoom), False)
         .dilate(pytweening.easeInSine)
         .time_rescale_exact(2, 's')
         .pop()
        )
