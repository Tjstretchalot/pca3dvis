"""Handles rendering the state. Uses matplotlib to do so."""
import typing
import numpy as np
import io
import pytypeutils as tus
import pympanim.acts as acts
import pca3dvis.state as state
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # pylint: disable=unused-import

def submask(pts: np.ndarray, mask: np.ndarray, mask2: np.ndarray):
    """Given that there exists some array arr where pts = arr[mask], this
    finds arr[mask2] where mask2 is some subset of mask.

    Args:
        pts (np.ndarray): arr[mask]
        mask (np.ndarray): some bool mask of the unknown array arr
        mask2 (np.ndarray): a subset of mask, i.e. a bool array that is false
            everywhere that mask is false but potentially false where mask is
            true

    Returns:
       arr[mask2]
    """
    tus.check_ndarrays(
        mask=(mask, ('samples',), 'bool'),
        mask2=(mask2, (('samples', mask.shape[0]),), 'bool'))
    shape = [mask.shape[0]]
    shape.extend(pts.shape[1:])
    arr = np.zeros(shape, dtype=pts.dtype)
    arr[mask] = pts
    return arr[mask2]

class ProjectedRenderer(acts.ActRenderer):
    """Renders a state.ProjectedState using matplotlib.

    Attributes:
        frame_size_in (tuple[float, float]): the frame size in inches
        dpi (int): the number of pixels per inch

        _frame_size (tuple[int, int]): the frame size in pixels, calculated
            from frame_size_in and dpi.
    """
    def __init__(self, frame_size_in: typing.Tuple[float, float], dpi: int):
        tus.check(
            frame_size_in=(frame_size_in, (list, tuple)),
            dpi=(dpi, int)
        )
        tus.check_listlike(frame_size_in=(frame_size_in, (int, float), 2))

        # this fixes rounding
        _frame_size = (
            int(round(frame_size_in[0] * dpi)),
            int(round(frame_size_in[1] * dpi))
        )
        frame_size_in = (_frame_size[0] / dpi, _frame_size[1] / dpi)

        self.frame_size_in = frame_size_in
        self.dpi = dpi
        self._frame_size = _frame_size

    @property
    def frame_size(self):
        return self._frame_size

    def _scatter(self, act_state, ax, pts, mask, skwargs):
        pts_handled = 0
        pts_to_handle = mask.sum()

        ind = -1
        while (ind < len(act_state.default_styling) - 1
               and pts_handled < pts_to_handle):
            ind += 1
            def_styling = act_state.default_styling[ind]
            overlap = mask * def_styling[0]
            osum = overlap.sum()
            if osum == 0:
                continue
            overlap_pts = submask(pts, mask, overlap)
            kwargs = def_styling[1].copy()
            kwargs.update(skwargs)
            if 'c' in kwargs:
                c = kwargs['c']
                if isinstance(c, np.ndarray):
                    kwargs['c'] = c[overlap]
            ax.scatter(overlap_pts[:, 0], overlap_pts[:, 1], overlap_pts[:, 2],
                       **kwargs)
            pts_handled += osum

    def render(self, act_state: state.ProjectedState) -> bytes:
        """Renders the given state to an rgba image encoded in raw format"""
        fig = self.render_mpl(act_state)
        fig.set_size_inches(*self.frame_size_in)
        hndl = io.BytesIO()
        fig.savefig(hndl, format='rgba', dpi=self.dpi)
        rawimg = hndl.getvalue()
        plt.close(fig)
        return rawimg

    def render_mpl(self, act_state: state.ProjectedState):
        """Returns a matplotlib figure with the state on it"""
        fig = plt.figure(figsize=self.frame_size_in)
        ax = fig.add_subplot(111, projection='3d')

        axtitle = ax.set_title(act_state.title)

        font_size = int((80 / 1920) * self._frame_size[0])
        axtitle.set_fontsize(font_size)

        renderer = fig.canvas.get_renderer()
        bb = axtitle.get_window_extent(renderer=renderer)
        while bb.width >= self._frame_size[0] * 0.9:
            font_size -= 5
            axtitle.set_fontsize(font_size)
            bb = axtitle.get_window_extent(renderer=renderer)

        for pts, mask, skwargs in act_state.visible_points:
            tus.check(skwargs=(skwargs, dict))
            self._scatter(act_state, ax, pts, mask, skwargs)

        ax.set_xlim(*[float(i) for i in act_state.zoom[0]])
        ax.set_ylim(*[float(i) for i in act_state.zoom[1]])
        ax.set_zlim(*[float(i) for i in act_state.zoom[2]])
        ax.view_init(*act_state.rotation)
        return fig
