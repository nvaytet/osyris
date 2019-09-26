# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2019 Osyris contributors (https://github.com/nvaytet/osyris)
# @author Neil Vaytet

import numpy as np
try:
    import ipyvolume as ipv
    disabled_3d = False
except ImportError:
    print("Warning: 3d plots are disabled because ipyvolume was not found.")
    disabled_3d = True

# from .plot import get_slice_direction
# from .plot_slice import plot_slice
from matplotlib import cm
from scipy.interpolate import griddata


def plot_volume(field, dx=0.0, dy=0.0, dz=0.0, fname=None, title=None,
                sinks=True, resolution=128, **kwargs):
    """
    Plot volume redering of 3D data cube.
    """

    if disabled_3d:
        print("plot_volume is disabled because ipyvolume is not installed.")
        return

    # Find parent container of object to plot
    holder = field.parent

    cube = np.where(np.logical_and(holder.get("x") >= -0.5*dx, \
                    np.logical_and(holder.get("x") <=  0.5*dx, \
                    np.logical_and(holder.get("y") >= -0.5*dy, \
                    np.logical_and(holder.get("y") <=  0.5*dy, \
                    np.logical_and(holder.get("z") >= -0.5*dz, \
                                   holder.get("z") <=  0.5*dz))))))

    V0, edges = np.histogramdd(np.array([holder.get("x")[cube],
                                         holder.get("y")[cube],
                                         holder.get("z")[cube]]).T,
                               bins=(resolution, resolution, resolution))
    V1, edges = np.histogramdd(np.array([holder.get("x")[cube],
                                         holder.get("y")[cube],
                                         holder.get("z")[cube]]).T,
                               bins=(resolution, resolution, resolution),
                               weights=field.get()[cube])
    V = V1 / V0
    ipv.quickvolshow(V)
    ipv.show()

    return


def plot_quiver(field, iskip=1, dx=0.0, dy=0.0, dz=0.0, fname=None,
                title=None, sinks=True, size=1, **kwargs):

    if disabled_3d:
        print("plot_quiver is disabled because ipyvolume is not installed.")
        return

    # Find parent container of object to plot
    holder = field.parent

    cube = np.where(np.logical_and(holder.get("x") >= -0.5*dx, \
                    np.logical_and(holder.get("x") <=  0.5*dx, \
                    np.logical_and(holder.get("y") >= -0.5*dy, \
                    np.logical_and(holder.get("y") <=  0.5*dy, \
                    np.logical_and(holder.get("z") >= -0.5*dz, \
                                   holder.get("z") <=  0.5*dz))))))

    ipv.figure()
    quiver = ipv.quiver(holder.get("x")[cube][::iskip],
                        holder.get("y")[cube][::iskip],
                        holder.get("z")[cube][::iskip],
                        field.x.get()[cube][::iskip],
                        field.y.get()[cube][::iskip],
                        field.z.get()[cube][::iskip], size=size)
    ipv.show()
    return




class slicer_3d:

    def __init__(self, field=None, direction="z", origin=[0, 0, 0], dx=0.0, dy=0.0, dz=0.0, resolution=129, interpolation="linear", lmax=0):

        # if disabled_3d:
        #     print("plot_slice3d is disabled because ipyvolume is not installed.")
        #     return

        from ipywidgets import FloatSlider, VBox
        holder = field.parent

        self.field = field
        self.direction = direction
        self.resolution = resolution
        self.interpolation = interpolation

        # Compute domain dimension for integration
        self.dx = dx
        if dy == 0.0:
            dy = dx
        self.dy = dy
        if dz == 0.0:
            dz = max(dx, dy)
        self.dz = dz
        # self.zmin = -0.5*self.dz
        # self.zmax = 0.5*self.dz
        self.nx = resolution
        self.ny = resolution
        self.nz = resolution
        self.origin = origin

        # Try to automatically determine lmax to speedup process
        if lmax == 0:
            dxlmax = holder.info["boxsize_scaled"] * (
                0.5**holder.info["levelmax_active"])
            target = min(dx, dy, dz) / float(resolution)
            lmax = round(np.log((min(dx, dy, dz) / float(resolution)) /
                                holder.info["boxsize_scaled"])/(np.log(0.5)))
        subset = np.where(
            np.logical_or(
                np.logical_and(holder.get("level", only_leafs=False) < lmax,
                               holder.get("leaf", only_leafs=False) > 0.0),
                holder.get("level", only_leafs=False) == lmax))

            # Distance from center
        dist2 = np.sqrt((holder.get("x", only_leafs=False)[subset]-origin[0])**2 +
                        (holder.get("y", only_leafs=False)[subset]-origin[1])**2 +
                        (holder.get("z", only_leafs=False)[subset]-origin[2])**2) - \
            np.sqrt(3.0)*0.5*holder.get("dx", only_leafs=False)[subset]

        # Select only the cells in contact with the slice., i.e. at a distance less than dx/2
        cube = np.where(np.abs(dist2) <= max(dx, dy, dz)*0.5*np.sqrt(3.0))

        datax = holder.get("x", only_leafs=False)[subset][cube] - origin[0]
        datay = holder.get("y", only_leafs=False)[subset][cube] - origin[1]
        dataz = holder.get("z", only_leafs=False)[subset][cube] - origin[2]

        self.xmin = -0.5 * self.dx
        self.xmax = self.xmin + self.dx
        self.ymin = -0.5 * self.dy
        self.ymax = self.ymin + self.dy
        self.zmin = -0.5 * self.dz
        self.zmax = self.zmin + self.dz
        dpx = (self.xmax-self.xmin)/float(self.nx)
        dpy = (self.ymax-self.ymin)/float(self.ny)
        dpz = (self.zmax-self.zmin)/float(self.nz)
        x = np.linspace(self.xmin+0.5*dpx, self.xmax-0.5*dpx, self.nx)
        y = np.linspace(self.ymin+0.5*dpy, self.ymax-0.5*dpy, self.ny)
        z = np.linspace(self.zmin+0.5*dpz, self.zmax-0.5*dpz, self.nz)
        self.grid_x, self.grid_y, self.grid_z = np.meshgrid(x, y, z)
        points = np.transpose([datax, datay, dataz])
        grids = (self.grid_x, self.grid_y, self.grid_z)

        # print("starting griddata with {} points".format(len(points)))
        # self.val = griddata(points, self.field.values[subset][cube], grids,
        #     method=interpolation)
        # print("griddata done")
        select = np.where(np.sqrt(
            (holder.get("x", only_leafs=False)[subset][cube] - self.grid_x)**2 +
            (holder.get("y", only_leafs=False)[subset][cube] - self.grid_y)**2 +
            (holder.get("z", only_leafs=False)[subset][cube] - self.grid_z)**2) < holder.get("dx", only_leafs=False)[subset][cube]*0.5)
        self.val = field.values[subset][cube][select]

        # from .interpolate import interpolate
        # print("starting interpolate with {} points".format(len(points)))
        # self.val = interpolate(self.field, points)
        # print("interpolate done")
        print(np.shape(self.val))


        # # self.lmax = lmax

        # # # Get direction vectors once and for all for the column_density.
        # # # This should be computed here and not inside the plot_slice routine as the origin
        # # # changes along the z direction inside the loop below.
        # # self.dx, self.dy, self.box, self.dir_vecs, self.origin = get_slice_direction(
        # #     holder, direction, dx, dy, origin=origin)

        # # Compute domain dimension for integration
        # self.dz = dz
        # if self.dz == 0.0:
        #     self.dz = max(self.dx, self.dy)
        # self.zmin = -0.5*self.dz
        # self.zmax = 0.5*self.dz
        # self.nx = resolution
        # self.ny = resolution
        # # if nz == 0:
        # #     nz = resolution
        # dpz = (zmax-zmin)/float(nz)
        # z = np.linspace(zmin+0.5*dpz, zmax-0.5*dpz, nz)

        # # We now create empty data arrays that will be filled by the cell data
        # z_scal = z_imag = z_cont = u_vect = v_vect = w_vect = u_strm = v_strm = w_strm = 0
        # if scalar:
        #     z_scal = np.zeros([ny, nx])
        # if image:
        #     z_imag = np.zeros([ny, nx])
        # if contour:
        #     z_cont = np.zeros([ny, nx])
        # if vec:
        #     u_vect = np.zeros([ny, nx])
        #     v_vect = np.zeros([ny, nx])
        #     w_vect = np.zeros([ny, nx])
        # if stream:
        #     u_strm = np.zeros([ny, nx])
        #     v_strm = np.zeros([ny, nx])
        #     w_strm = np.zeros([ny, nx])

        # # Define equation of a plane
        # a_plane = dir_vecs[0][1][0]
        # b_plane = dir_vecs[0][1][1]
        # c_plane = dir_vecs[0][1][2]
        # d_plane = -dir_vecs[0][1][0]*origin[0] - \
        #     dir_vecs[0][1][1]*origin[1]-dir_vecs[0][1][2]*origin[2]





        # self.norm = cm.colors.Normalize(vmax=field.get().max(), vmin=field.get().min())
        self.colormap = cm.viridis
        # color = self.colormap(self.norm(self.field.get()))

        ipv.figure()
        # x = holder.get("x")
        # y = holder.get("y")
        # z = holder.get("z")


        # [x, y, z_scal_slice, z_imag_slice, z_cont_slice, u_vect_slice, v_vect_slice,
        #         w_vect_slice, u_strm_slice, v_strm_slice, w_strm_slice] = \
        #         plot_slice(scalar=self.field,
        #                    direction=self.direction, dx=self.dx, dy=self.dy, copy=True, resolution=self.resolution,
        #                    origin=self.origin, plot=False, interpolation=self.interpolation, lmax=self.lmax,
        #                    slice_direction=[self.dx, self.dy, self.box, self.dir_vecs, self.origin])

        # self.norm = cm.colors.Normalize(vmax=field.get().max(), vmin=field.get().min())
        self.norm = cm.colors.Normalize(vmax=np.nanmax(val), vmin=np.nanmin(val))
        # self.colormap = cm.coolwarm
        # color = self.colormap(self.norm(val))

        # print(z_scal_slice.max(), z_scal_slice.min())

                # Make outline
        x1, y1 = np.meshgrid([x.min(), x.max()], [y.min(), y.max()])
        w1 = ipv.plot_wireframe(x1, y1, np.ones_like(x1) * self.zmin, color="black")
        w2 = ipv.plot_wireframe(x1, y1, np.ones_like(x1) * self.zmax, color="black")

        print(np.shape(x))
        print(np.shape(y))
        print(np.shape(z))
        print(np.shape(val))

        color = self.colormap(self.norm(self.val[:,:,nz//2].flatten()))
        # self.xx, self.yy = np.meshgrid(x, y)
        self.surf = ipv.plot_surface(self.grid_x, self.grid_y, np.zeros_like(self.grid_x), color=color[...,:3])

        # direction = "yxz"
        # self.dx, self.dy, self.box, self.dir_vecs, self.origin = get_slice_direction(
        #     holder, direction, dx, dy, origin=origin)

        # [x, y, z_scal_slice, z_imag_slice, z_cont_slice, u_vect_slice, v_vect_slice,
        #         w_vect_slice, u_strm_slice, v_strm_slice, w_strm_slice] = \
        #         plot_slice(scalar=self.field,
        #                    direction=direction, dx=self.dx, dy=self.dy, copy=True, resolution=self.resolution,
        #                    origin=self.origin, plot=False, interpolation=self.interpolation, lmax=self.lmax,
        #                    slice_direction=[self.dx, self.dy, self.box, self.dir_vecs, self.origin])
        # color1 = self.colormap(self.norm(z_scal_slice.flatten()))
        # xx, yy = np.meshgrid(x, y)
        # self.surf1 = ipv.plot_surface(xx, np.zeros_like(z_scal_slice), yy, color=color1[...,:3])

        # X = xx.copy().flatten()
        self.size = IntSlider(value=nz//2, min=0, max=self.nz)

        self.size.observe(self.update_z, names="value")
        display(VBox([ipv.gcc(), self.size]))
        return


    def update_z(self, change):
        # global X, surf, colormap, norm
        # print(change["new"])
        # print()
        self.surf.z = np.ones_like(self.surf.x) * change["new"]
        # a = np.sin(np.sqrt(((xx-c)/b)**2 + ((yy-c)/b)**2 + ((z[change["new"]]-d)/b)**2))
        #color = colormap(norm(a[:,:,change["new"]].flatten()))
        [x, y, z_scal_slice, z_imag_slice, z_cont_slice, u_vect_slice, v_vect_slice,
            w_vect_slice, u_strm_slice, v_strm_slice, w_strm_slice] = \
            plot_slice(scalar=self.field,
                       direction=self.direction, dx=self.dx, dy=self.dy, copy=True, resolution=self.resolution,
                       origin=[0, 0, change["new"]], plot=False, interpolation=self.interpolation, lmax=self.lmax,
                       slice_direction=[self.dx, self.dy, self.box, self.dir_vecs, [0, 0, change["new"]]])
        self.norm = cm.colors.Normalize(vmax=np.nanmax(z_scal_slice), vmin=np.nanmin(z_scal_slice))
        color = self.colormap(self.norm(z_scal_slice.flatten()))
        self.surf.color = color[..., :3]
        return



def plot_slice3d(field=None, direction="z", origin=[0, 0, 0], dx=0.0, dy=0.0, dz=0.0, resolution=128, interpolation="linear", lmax=0):

    return slicer_3d(field=field, direction=direction, origin=origin, dx=dx, dy=dy, dz=dz, resolution=resolution, interpolation=interpolation, lmax=lmax)
