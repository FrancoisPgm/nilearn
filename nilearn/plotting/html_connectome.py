import json

import numpy as np
from scipy import sparse
from io import BytesIO

from nilearn._utils import rename_parameters, check_niimg_4d
from nilearn._utils.niimg import _safe_get_data
from nilearn.image.resampling import coord_transform
from .. import datasets
from . import cm

from .html_stat_map import _bytesIO_to_base64
from .js_plotting_utils import (add_js_lib, mesh_to_plotly,
                                encode, colorscale, get_html_template,
                                to_color_strings)
from nilearn.reporting import HTMLDocument


class ConnectomeView(HTMLDocument):
    pass


def _prepare_line(edges, nodes):
    path_edges = np.zeros(len(edges) * 3, dtype=int)
    path_edges[::3] = edges
    path_edges[1::3] = edges
    path_nodes = np.zeros(len(nodes) * 3, dtype=int)
    path_nodes[::3] = nodes[:, 0]
    path_nodes[1::3] = nodes[:, 1]
    return path_edges, path_nodes


def _get_connectome(adjacency_matrix, coords, threshold=None,
                    marker_size=None, cmap=cm.blue_red, symmetric_cmap=True):
    connectome = {}
    coords = np.asarray(coords, dtype='<f4')
    adjacency_matrix = np.nan_to_num(adjacency_matrix, copy=True)
    colors = colorscale(
        cmap, adjacency_matrix.ravel(), threshold=threshold,
        symmetric_cmap=symmetric_cmap)
    connectome['colorscale'] = colors['colors']
    connectome['cmin'] = float(colors['vmin'])
    connectome['cmax'] = float(colors['vmax'])
    if threshold is not None:
        adjacency_matrix[
            np.abs(adjacency_matrix) <= colors['abs_threshold']] = 0
    s = sparse.coo_matrix(adjacency_matrix)
    nodes = np.asarray([s.row, s.col], dtype=int).T
    edges = np.arange(len(nodes))
    path_edges, path_nodes = _prepare_line(edges, nodes)
    connectome["_con_w"] = encode(np.asarray(s.data, dtype='<f4')[path_edges])
    c = coords[path_nodes]
    if np.ndim(marker_size) > 0:
        marker_size = np.asarray(marker_size)
        marker_size = marker_size[path_nodes]
    x, y, z = c.T
    for coord, cname in [(x, "x"), (y, "y"), (z, "z")]:
        connectome["_con_{}".format(cname)] = encode(
            np.asarray(coord, dtype='<f4'))
    connectome["markers_only"] = False
    if hasattr(marker_size, 'tolist'):
        marker_size = marker_size.tolist()
    connectome['marker_size'] = marker_size
    return connectome

def _get_volume(img, threshold=0, stride=1, t_start=0, t_end=-1, n_t=50,
                marker_size=3, cmap=cm.cold_hot, symmetric_cmap=True):
    connectome = {}
    img = check_niimg_4d(img)
    if t_end < 0:
        t_end = img.shape[3] + t_end
    t_idx = np.round(np.linspace(t_start, t_end, n_t)).astype(int)
    data = _safe_get_data(img)[::stride,::stride,::stride,t_idx]
    mask = np.abs(data[:,:,:,0]) > threshold
    i, j, k = mask.nonzero()
    x, y, z = coord_transform(i*stride, j*stride, k*stride, img.affine)
    for coord, cname in [(x, "x"), (y, "y"), (z, "z")]:
        connectome["_con_{}".format(cname)] = encode(
            np.asarray(coord, dtype='<f4'))
    colors = colorscale(cmap, data.ravel(),
                        symmetric_cmap=symmetric_cmap)
    connectome['colorscale'] = colors['colors']
    connectome['cmin'] = float(colors['vmin'])
    connectome['cmax'] = float(colors['vmax'])
    connectome['n_time'] = n_t
    values = [encode(np.asarray(data[i,j,k,t], dtype='<f4')) for t in range(data.shape[3])]
    connectome['values'] = values

    return connectome

def _get_markers(coords, colors):
    connectome = {}
    coords = np.asarray(coords, dtype='<f4')
    x, y, z = coords.T
    for coord, cname in [(x, "x"), (y, "y"), (z, "z")]:
        connectome["_con_{}".format(cname)] = encode(
            np.asarray(coord, dtype='<f4'))
    connectome["marker_color"] = to_color_strings(colors)
    connectome["markers_only"] = True
    return connectome


def _make_connectome_html(connectome_info, embed_js=True):
    plot_info = {"connectome": connectome_info}
    mesh = datasets.fetch_surf_fsaverage()
    for hemi in ['pial_left', 'pial_right']:
        plot_info[hemi] = mesh_to_plotly(mesh[hemi])
    as_json = json.dumps(plot_info)
    as_html = get_html_template(
        'connectome_plot_template.html').safe_substitute(
            {'INSERT_CONNECTOME_JSON_HERE': as_json,
             'INSERT_PAGE_TITLE_HERE': (
                 connectome_info["title"] or "Connectome plot")})
    as_html = add_js_lib(as_html, embed_js=embed_js)
    return ConnectomeView(as_html)


def view_connectome(adjacency_matrix, node_coords, edge_threshold=None,
                    edge_cmap=cm.bwr, symmetric_cmap=True,
                    linewidth=6., node_size=3., colorbar=True,
                    colorbar_height=.5, colorbar_fontsize=25,
                    title=None, title_fontsize=25):
    """
    Insert a 3d plot of a connectome into an HTML page.

    Parameters
    ----------
    adjacency_matrix : ndarray, shape=(n_nodes, n_nodes)
        the weights of the edges.

    node_coords : ndarray, shape=(n_nodes, 3)
        the coordinates of the nodes in MNI space.

    edge_threshold : str, number or None, optional (default=None)
        If None, no thresholding.
        If it is a number only connections of amplitude greater
        than threshold will be shown.
        If it is a string it must finish with a percent sign,
        e.g. "25.3%", and only connections of amplitude above the
        given percentile will be shown.

    edge_cmap : str or matplotlib colormap, optional

    symmetric_cmap : bool, optional (default=True)
        Make colormap symmetric (ranging from -vmax to vmax).

    linewidth : float, optional (default=6.)
        Width of the lines that show connections.

    node_size : float, optional (default=3.)
        Size of the markers showing the seeds in pixels.

    colorbar : bool, optional (default=True)
        add a colorbar

    colorbar_height : float, optional (default=.5)
        height of the colorbar, relative to the figure height

    colorbar_fontsize : int, optional (default=25)
        fontsize of the colorbar tick labels

    title : str, optional (default=None)
        title for the plot

    title_fontsize : int, optional (default=25)
        fontsize of the title

    Returns
    -------
    ConnectomeView : plot of the connectome.
        It can be saved as an html page or rendered (transparently) by the
        Jupyter notebook. Useful methods are :

        - 'resize' to resize the plot displayed in a Jupyter notebook
        - 'save_as_html' to save the plot to a file
        - 'open_in_browser' to save the plot and open it in a web browser.

    See Also
    --------
    nilearn.plotting.plot_connectome:
        projected views of a connectome in a glass brain.

    nilearn.plotting.view_markers:
        interactive plot of colored markers

    nilearn.plotting.view_surf, nilearn.plotting.view_img_on_surf:
        interactive view of statistical maps or surface atlases on the cortical
        surface.

    """
    connectome_info = _get_connectome(
        adjacency_matrix, node_coords,
        threshold=edge_threshold, cmap=edge_cmap,
        symmetric_cmap=symmetric_cmap, marker_size=node_size)
    connectome_info['line_width'] = linewidth
    connectome_info['colorbar'] = colorbar
    connectome_info['cbar_height'] = colorbar_height
    connectome_info['cbar_fontsize'] = colorbar_fontsize
    connectome_info['title'] = title
    connectome_info['title_fontsize'] = title_fontsize
    return _make_connectome_html(connectome_info)


def view_markers(marker_coords, marker_color=None, marker_size=5.,
                 title=None, title_fontsize=25):
    """
    Insert a 3d plot of markers in a brain into an HTML page.

    Parameters
    ----------
    marker_coords : ndarray, shape=(n_nodes, 3)
        the coordinates of the nodes in MNI space.

    marker_color : ndarray, shape=(n_nodes,)
        colors of the markers: list of strings, hex rgb or rgba strings, rgb
        triplets, or rgba triplets (i.e. formats accepted by matplotlib, see
        https://matplotlib.org/users/colors.html#specifying-colors)

    marker_size : float or array-like, optional (default=3.)
        Size of the markers showing the seeds in pixels.

    title : str, optional (default=None)
        title for the plot

    title_fontsize : int, optional (default=25)
        fontsize of the title

    Returns
    -------
    ConnectomeView : plot of the markers.
        It can be saved as an html page or rendered (transparently) by the
        Jupyter notebook. Useful methods are :

        - 'resize' to resize the plot displayed in a Jupyter notebook
        - 'save_as_html' to save the plot to a file
        - 'open_in_browser' to save the plot and open it in a web browser.

    See Also
    --------
    nilearn.plotting.plot_connectome:
        projected views of a connectome in a glass brain.

    nilearn.plotting.view_connectome:
        interactive plot of a connectome.

    nilearn.plotting.view_surf, nilearn.plotting.view_img_on_surf:
        interactive view of statistical maps or surface atlases on the cortical
        surface.

    """
    if marker_color is None:
        marker_color = ['red' for i in range(len(marker_coords))]
    connectome_info = _get_markers(marker_coords, marker_color)
    if hasattr(marker_size, 'tolist'):
        marker_size = marker_size.tolist()
    connectome_info["marker_size"] = marker_size
    connectome_info['title'] = title
    connectome_info['title_fontsize'] = title_fontsize
    return _make_connectome_html(connectome_info)


def view_volume(img, threshold=0, stride=1, t_start=0, t_end=-1, n_t=50,
                marker_size=3., opacity=0.8, cmap="plasma", symmetric_cmap=True,
                colorbar=True, colorbar_height=.5, colorbar_fontsize=11,
                title=None, title_fontsize=16):
    """
    Insert a 4d plot of a brain into an HTML page.

    Parameters
    ----------
    img : Nifti image of a time-series.

    marker_size : float or array-like, optional (default=3.)
        Size of the markers showing the seeds in pixels.

    cmpa :

    symmetric_cmap :

    colorbar : bool, optional (default=True)
        add a colorbar

    colorbar_height : float, optional (default=.5)
        height of the colorbar, relative to the figure height

    colorbar_fontsize : int, optional (default=25)
        fontsize of the colorbar tick labels

    title : str, optional (default=None)
        title for the plot

    title_fontsize : int, optional (default=25)
        fontsize of the title

    Returns
    -------
    ConnectomeView : plot of the markers.
        It can be saved as an html page or rendered (transparently) by the
        Jupyter notebook. Useful methods are :

        - 'resize' to resize the plot displayed in a Jupyter notebook
        - 'save_as_html' to save the plot to a file
        - 'open_in_browser' to save the plot and open it in a web browser.

    See Also
    --------
    nilearn.plotting.plot_connectome:
        projected views of a connectome in a glass brain.

    nilearn.plotting.view_connectome:
        interactive plot of a connectome.

    nilearn.plotting.view_surf, nilearn.plotting.view_img_on_surf:
        interactive view of statistical maps or surface atlases on the cortical
        surface.

    """

    connectome_info = _get_volume(img, threshold, stride, t_start, t_end, n_t,
                                  marker_size, cmap, symmetric_cmap)
    connectome_info["4D"] = True
    connectome_info["marker_size"] = marker_size
    connectome_info['title'] = title
    connectome_info['title_fontsize'] = title_fontsize
    connectome_info['opacity'] = opacity
    connectome_info['colorbar'] = colorbar
    connectome_info['cbar_height'] = colorbar_height
    connectome_info['cbar_fontsize'] = colorbar_fontsize
    return _make_connectome_html(connectome_info)
