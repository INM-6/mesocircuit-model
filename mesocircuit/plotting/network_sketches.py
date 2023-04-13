# consider making it standalone in the future
from mesocircuit.parameterization.base_plotting_params import plot_dict
import numpy as np
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, RegularPolygon, FancyArrowPatch
import mpl_toolkits.mplot3d.art3d as art3d
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
matplotlib.use('Agg')

matplotlib.rcParams.update(plot_dict['rcParams'])

# https://gist.github.com/WetHat/1d6cd0f7309535311a539b42cccca89c


class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)


setattr(Axes3D, 'arrow3D', _arrow3D)


def plot_mesocircuit_icon(gs, elev=12, azim=-50, scale_fs=0.7):
    """
    Plots a schematic network icon to gridspec cell.

    Parameters
    ----------
    gs
        A gridspec cell to plot into.
    elev
        Elevation angle in z-plane.
    azim
        Azimuth angle in x,y-plane.
    scale_fs
        Scaling factor for font size.
    """
    ax = plt.subplot(gs, projection='3d', computed_zorder=False)

    pop_colors = plot_dict['pop_colors']
    pop_labels = plot_dict['pop_labels']
    zcnt = 0
    for i, col in enumerate(pop_colors[::-1]):
        if i == 0:  # TC
            patch = Circle((0.5, 0.5), 0.1, facecolor=col)
            z = -5 / (len(pop_colors) - 1)
            xshift = 0.4
            yshift = 0.4
        else:
            patch = Rectangle((0, 0), 1, 1, facecolor=col)
            z = 1.5*i / (len(pop_colors) - 1) + zcnt
            xshift = -0.02
            yshift = -0.02
            if not i % 2:
                zcnt += 2 / (len(pop_colors) - 1)
        ax.add_patch(patch)
        art3d.pathpatch_2d_to_3d(patch, z=z, zdir="z")

        if i % 2:
            zshift = 0.
        else:
            zshift = 0.07
        if i == 0:
            zshift = -0.05
        ax.text(1. - xshift, 1 - yshift, z+zshift, pop_labels[::-1][i],
                fontsize=matplotlib.rcParams['font.size'] * scale_fs,
                verticalalignment='center')
    ax.text(1, 1, 2.3, '4 mm', 'x',
            fontsize=matplotlib.rcParams['font.size'] * scale_fs,
            horizontalalignment='right')
    ax.text(0, 0, 2.3, '4 mm', 'y',
            fontsize=matplotlib.rcParams['font.size'] * scale_fs,
            horizontalalignment='left')

    # exponential profile indicating connectivity
    xctr = 0.5
    yctr = 0.5
    X = np.arange(xctr-0.2, xctr+0.2, 0.01)
    Y = np.arange(yctr - 0.2, yctr+0.2, 0.01)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt((X-xctr)**2 + (Y-yctr)**2)
    Z = z + 0.25 * np.exp(-R/(0.2/4))

    # make bottom of surface round
    for i in np.arange(np.shape(Z)[0]):
        for j in np.arange(np.shape(Z)[1]):
            if ((X[i, j]-xctr)**2 + (Y[i, j]-yctr)**2) > 0.15**2:
                Z[i, j] = np.nan

    ax.plot_surface(X, Y, Z, cmap='bone')

    ax.grid(False)
    ax.view_init(elev=elev, azim=azim)
    plt.axis('off')
    return


def plot_mesocircuit_icon2(gs, elev=12, azim=-50, scale_fs=0.7):
    """
    Plots a schematic network icon to gridspec cell.

    Parameters
    ----------
    gs
        A gridspec cell to plot into.
    elev
        Elevation angle in z-plane.
    azim
        Azimuth angle in x,y-plane.
    scale_fs
        Scaling factor for font size.
    """

    ax = plt.subplot(gs, projection='3d', computed_zorder=False)

    pop_colors = plot_dict['pop_colors']
    type_colors = plot_dict['type_colors']
    pop_labels = plot_dict['pop_labels']
    layer_labels = plot_dict['layer_labels']

    ctr_exc = [0.3, 0.2]
    ctr_inh = [0.7, 0.2]

    mutation_scale = 10

    # TC
    tc_layer = Circle(xy=(0.5, 0.5), radius=0.1,
                      facecolor='white', edgecolor='k')
    ax.add_patch(tc_layer)
    art3d.pathpatch_2d_to_3d(tc_layer, z=-1.5, zdir="z")
    # TC neurons
    tc = RegularPolygon(xy=(0.5, 0.5), radius=0.1, numVertices=3, orientation=0,
                        facecolor=pop_colors[-1],
                        edgecolor='k')
    ax.add_patch(tc)
    art3d.pathpatch_2d_to_3d(tc, z=-1.5, zdir="z")

    # cortex
    for z, ll in enumerate(layer_labels[::-1]):

        # layer
        layer = Rectangle(xy=(0, 0), width=1, height=1,
                          facecolor='white', edgecolor='k')
        ax.add_patch(layer)
        art3d.pathpatch_2d_to_3d(layer, z=z, zdir="z")

        # excitatory neurons
        exc = RegularPolygon(xy=ctr_exc, radius=0.1, numVertices=3, orientation=0,
                             facecolor=pop_colors[::-1][2 + 2*z],
                             edgecolor='k')
        ax.add_patch(exc)
        art3d.pathpatch_2d_to_3d(exc, z=z, zdir="z")

        # inhibitory neurons
        inh = Circle(xy=ctr_inh, radius=0.08,
                     facecolor=pop_colors[::-1][1 + 2*z],
                     edgecolor='k')
        ax.add_patch(inh)
        art3d.pathpatch_2d_to_3d(inh, z=z, zdir="z")

        # excitatory connections
        ax.arrow3D(x=ctr_exc[0]+0.02, y=0.25, z=z,
                   dx=0.34, dy=0, dz=0,
                   mutation_scale=mutation_scale,
                   arrowstyle="->",
                   color=type_colors[0])
        draw_edge_arrow(ax=ax, x=ctr_exc[0]-0.07, y=ctr_exc[1], z=z,
                        xshift1=-0.06, yshift1=0.2,
                        xshift2=0.13, yshift2=-0.1,
                        color=type_colors[0])

        # inhibitory connections
        ax.arrow3D(x=ctr_inh[0]-0.07, y=0.2, z=z,
                   dx=-0.29, dy=0, dz=0,
                   mutation_scale=mutation_scale,
                   arrowstyle="->",
                   color=type_colors[1])
        draw_edge_arrow(ax=ax, x=ctr_inh[0]+0.08, y=ctr_exc[1], z=z,
                        xshift1=+0.05, yshift1=0.2,
                        xshift2=-0.13, yshift2=-0.12,
                        color=type_colors[1])

    # ax.grid(False)
    #ax.view_init(elev=elev, azim=azim)
    ax.set_zlim(top=3)

    ax.view_init(elev=20, azim=-50)
    #ax.view_init(elev=45, azim=0)
    plt.axis('off')
    return


def draw_edge_arrow(ax, x, y, z, xshift1, yshift1, xshift2, yshift2, color='k', mutation_scale=10):
    # xshift
    ax.arrow3D(x=x, y=y, z=z,
               dx=xshift1, dy=0, dz=0,
               mutation_scale=mutation_scale,
               arrowstyle="-",
               shrinkA=0, shrinkB=0,
               color=color)
    # yshift
    ax.arrow3D(x=x+xshift1, y=y, z=z,
               dx=0, dy=yshift1, dz=0,
               mutation_scale=mutation_scale,
               arrowstyle="-",
               shrinkA=0, shrinkB=0,
               color=color)
    # xshift back
    ax.arrow3D(x=x+xshift1, y=y+yshift1, z=z,
               dx=xshift2, dy=0, dz=0,
               mutation_scale=mutation_scale,
               arrowstyle="-",
               shrinkA=0, shrinkB=0,
               color=color)
    # yshift back
    ax.arrow3D(x=x+xshift1+xshift2, y=y+yshift1, z=z,
               dx=0, dy=yshift2, dz=0,
               mutation_scale=mutation_scale,
               arrowstyle="->",
               shrinkA=0, shrinkB=0,
               color=color)


def plot_mesocircuit_icon3(gs, type='upscaled'):
    """
    Plots a schematic network icon to gridspec cell.

    Parameters
    ----------
    gs
        A gridspec cell to plot into.
    type
        'reference' or 'upscaled'
    """
    np.random.seed(1242)

    # num_neurons_1mm2_SvA2018
    num_neurons = np.array(
        [47386, 13366, 70387, 17597, 20740, 4554, 19839, 4063])
    layer_sizes = [num_neurons[i] + num_neurons[i+1]
                   for i in np.arange(8, step=2)]
    # scaled
    layer_sizes /= np.max(layer_sizes)

    ax = plt.subplot(gs, projection='3d', computed_zorder=False)

    pop_colors = plot_dict['pop_colors']
    type_colors = plot_dict['type_colors']
    layer_labels = plot_dict['layer_labels']

    ctr_exc = [0.3, 0.2]
    ctr_inh = [0.7, 0.2]

    conn_ctr = [0.8, 0.6]

    mutation_scale = 10

    # cortex
    z = 0
    z_lctrs = np.zeros(4)  # layer centers
    for i, ll in enumerate(layer_labels[::-1]):

        z_ctr = z + layer_sizes[::-1][i] / 2 - 0.04
        z_lctrs[i] = z_ctr

        layer_size = layer_sizes[::-1][i]

        # layer
        layer = Rectangle(xy=(0, 0), width=1, height=1,
                          facecolor='white', edgecolor='k')
        ax.add_patch(layer)
        art3d.pathpatch_2d_to_3d(layer, z=z, zdir="z")

        # neurons
        num_neurons_exc = int(num_neurons[::-1][1 + 2*i] / 100)
        num_neurons_inh = int(num_neurons[::-1][2*i] / 100)
        pos_x_exc = 0.02 + 0.96*np.random.rand(num_neurons_exc)
        pos_y_exc = 0.02 + 0.96*np.random.rand(num_neurons_exc)
        pos_x_inh = 0.02 + 0.96*np.random.rand(num_neurons_inh)
        pos_y_inh = 0.02 + 0.96*np.random.rand(num_neurons_inh)
        ax.scatter(xs=pos_x_exc, ys=pos_y_exc, zs=z,
                   marker=',',
                   s=matplotlib.rcParams['lines.markersize'] * 0.01,
                   color=pop_colors[::-1][2 + 2*i], alpha=1)
        ax.scatter(xs=pos_x_inh, ys=pos_y_inh, zs=z,
                   marker=',',
                   s=matplotlib.rcParams['lines.markersize'] * 0.01,
                   color=pop_colors[::-1][1 + 2*i], alpha=1)

        if i == 3:
            conns = RegularPolygon(xy=conn_ctr, radius=0.1, numVertices=3, orientation=12.05,
                                   facecolor=pop_colors[::-1][2 + 2*i],
                                   edgecolor='k')
            ax.add_patch(conns)
            art3d.pathpatch_2d_to_3d(conns, z=z, zdir="z")

        # exponential profile indicating connectivity
        if i == 2 and type == 'upscaled':
            xctr = conn_ctr[0]
            yctr = conn_ctr[1]
            X = np.arange(xctr-0.2, xctr+0.2, 0.01)
            Y = np.arange(yctr - 0.2, yctr+0.2, 0.01)
            X, Y = np.meshgrid(X, Y)
            R = np.sqrt((X-xctr)**2 + (Y-yctr)**2)
            Z = z + layer_size * np.exp(-R/(0.06))

            # make bottom of surface round
            for k in np.arange(np.shape(Z)[0]):
                for l in np.arange(np.shape(Z)[1]):
                    if ((X[k, l]-xctr)**2 + (Y[k, l]-yctr)**2) > 0.16**2:
                        Z[k, l] = np.nan

            ax.plot_surface(X, Y, Z,  rstride=1, cstride=1, shade=True,  # cmap='bone',
                            antialiased=False, color=pop_colors[::-1][2 + 2*i])

        if i == 2 and type == 'reference':
            for j in np.arange(20):
                ax.plot(xs=[pos_x_exc[j], conn_ctr[0]],
                        ys=[pos_y_exc[j], conn_ctr[1]],
                        zs=[z, z+layer_size],
                        color=pop_colors[::-1][2+2*i],
                        zorder=1)

        # front
        front = Rectangle(xy=(0, z), width=1, height=layer_sizes[::-1][i],
                          facecolor='white', edgecolor='k')
        ax.add_patch(front)
        art3d.pathpatch_2d_to_3d(front, z=0, zdir="y")

        # excitatory neurons
        exc = RegularPolygon(xy=(ctr_exc[0], z_ctr-0.025), radius=0.1, numVertices=3, orientation=0,
                             facecolor=pop_colors[::-1][2 + 2*i],
                             edgecolor='k')
        ax.add_patch(exc)
        art3d.pathpatch_2d_to_3d(exc, z=0, zdir="y")

        # inhibitory neurons
        inh = Circle(xy=(ctr_inh[0], z_ctr), radius=0.08,
                     facecolor=pop_colors[::-1][1 + 2*i],
                     edgecolor='k')
        ax.add_patch(inh)
        art3d.pathpatch_2d_to_3d(inh, z=0, zdir="y")

        # excitatory connections
        ax.arrow3D(x=ctr_exc[0]+0.02, y=0, z=z_ctr+0.025,
                   dx=0.32, dy=0, dz=0,
                   mutation_scale=mutation_scale,
                   arrowstyle="->",
                   color=type_colors[0])

        draw_edge_arrow_xzplane(ax=ax, x=ctr_exc[0]-0.05, y=0, z=z_ctr,
                                xshift1=-0.08, zshift1=0.16,
                                xshift2=0.13, zshift2=-0.08,
                                color=type_colors[0])

        # inhibitory connections
        ax.arrow3D(x=ctr_inh[0]-0.07, y=0, z=z_ctr-0.025,
                   dx=-0.29, dy=0, dz=0,
                   mutation_scale=mutation_scale,
                   arrowstyle="->",
                   color=type_colors[1])

        draw_edge_arrow_xzplane(ax=ax, x=ctr_inh[0]+0.09, y=0, z=z_ctr,
                                xshift1=+0.04, zshift1=0.16,
                                xshift2=-0.13, zshift2=-0.08,
                                color=type_colors[1])

        # layer annotations
        ax.text(x=1.-0.02, y=0, z=z + layer_sizes[::-1][i]-0.02, zdir='x',
                s=ll,
                horizontalalignment='right', verticalalignment='top')

        z += layer_sizes[::-1][i]

    # layer crossing connections
    # reverse layer center such that top layer (2/3) has index 0
    z_lctrs = z_lctrs[::-1]
    # L2/3E -> L5E
    draw_edge_arrow_xzplane(ax=ax, x=ctr_exc[0]-0.08, y=0, z=z_lctrs[0]-0.05,
                            xshift1=-0.1, zshift1=-1.49,
                            xshift2=0.1, zshift2=0,
                            color=type_colors[0])

    # ax.grid(False)
    #ax.view_init(elev=elev, azim=azim)
    ax.set_zlim(bottom=0, top=np.sum(layer_sizes))
    ax.set_box_aspect(aspect=(1, 1, 2))

    ax.view_init(elev=20, azim=-50)
    #ax.view_init(elev=90, azim=0)
    #ax.view_init(elev=20, azim=-90)
    #ax.view_init(elev=45, azim=0)
    plt.axis('off')

    # TODO
    # other connections to front
    # illustration of distance dependence at back
    # 4x4 mm^2 and 1 mm^2
    # consider TC
    # random dots in population colors
    return


def draw_edge_arrow_xzplane(
        ax, x, y, z, xshift1, zshift1, xshift2, zshift2, color='k', mutation_scale=10):
    # xshift
    ax.arrow3D(x=x, y=y, z=z,
               dx=xshift1, dy=0, dz=0,
               mutation_scale=mutation_scale,
               arrowstyle='-',
               shrinkA=0, shrinkB=0,
               color=color)
    # zshift
    ax.arrow3D(x=x+xshift1, y=y, z=z,
               dx=0, dy=0, dz=zshift1,
               mutation_scale=mutation_scale,
               arrowstyle='-',
               shrinkA=0, shrinkB=0,
               color=color)

    if zshift2 == 0:
        arrowstyle = '->'
    else:
        arrowstyle = '-'

    # xshift back
    ax.arrow3D(x=x+xshift1, y=y, z=z+zshift1,
               dx=xshift2, dy=0, dz=0,
               mutation_scale=mutation_scale,
               arrowstyle=arrowstyle,
               shrinkA=0, shrinkB=0,
               color=color)
    # zshift back
    ax.arrow3D(x=x+xshift1+xshift2, y=y, z=z+zshift1,
               dx=0, dy=0, dz=zshift2,
               mutation_scale=mutation_scale,
               arrowstyle='->',
               shrinkA=0, shrinkB=0,
               color=color)


def mesocircuit_icon():
    """
    """
    # print('Plotting.')

    fig = plt.figure(figsize=(5, 5))
    gs = gridspec.GridSpec(1, 1)
    #gs.update(left=0.1, right=0.98, bottom=0.05, top=0.98)
    plot_mesocircuit_icon3(gs[0], type='reference')

    plt.savefig('mesocircuit_icon.pdf')
    return


if __name__ == '__main__':
    mesocircuit_icon()
