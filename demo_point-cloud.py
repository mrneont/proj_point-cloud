import sys, copy

import numpy                 as np
import matplotlib.pyplot     as plt
import mpl_toolkits.mplot3d  as m3d

import lib_point_cloud       as lpc

# -----------------------------------------------------------------------
# An example script for using the small 'point-clouds' library.
#
# Create some 3D points, and then visualize them along with some
# relevant axes and very simple outlier conditions.  Estimate some
# summary properties of them, including spread in meaningful axes for
# that cloud.
# 
# written by PA Taylor (SSCC, NIMH, NIH)
#
# -----------------------------------------------------------------------

if __name__ == "__main__" :

    # ---------------- generate a point-cloud ----------------
    N   = 100
    pts = lpc.make_test_pts(N, frac_out = 0.1)

    # ---------------- calc point-cloud outliers & axes ----------------

    # get fundamental properties of point-cloud:
    #   mu       = the coordinate of the mean of the point-cloud
    #   eig_axes = the slopes/axes of the point-cloud
    mu, eig_axes, cov, U, S = lpc.calc_slope_int_pts(pts)

    mdist = lpc.calc_mahalanobis_dist(pts, mu, cov)

    # see if there are any outliers
    #   pts_in   = point-cloud of non-outliers (must be non-empty)
    #   pts_out  = point-cloud of outliers (might be empty)
    filt, pts_in, pts_out = lpc.filter_no_iqr_out(pts, mu, eig_axes)

    Nin  = np.shape(pts_in)[0]                # number of non-outliers
    Nout = np.shape(pts_out)[0]               # number of outliers

    # calc axis bounds (= the pts*bnd* var) for various point-clouds:
    # all, non-outliers, and outliers
    pts0, pts_bnd0 = lpc.calc_bnds_along_line(pts, mu, eig_axes, 0)
    pts1, pts_bnd1 = lpc.calc_bnds_along_line(pts, mu, eig_axes, 1)
    pts2, pts_bnd2 = lpc.calc_bnds_along_line(pts, mu, eig_axes, 2)
    if Nin :
        # first need to calc mean+axes of this new point-cloud (well,
        # could be same as original)
        mu_in, eig_axes_in, cov, U, S = lpc.calc_slope_int_pts(pts_in)
        pts_in0, pts_in_bnd0 = lpc.calc_bnds_along_line(pts_in, mu_in, eig_axes_in, 0)
        pts_in1, pts_in_bnd1 = lpc.calc_bnds_along_line(pts_in, mu_in, eig_axes_in, 1)
        pts_in2, pts_in_bnd2 = lpc.calc_bnds_along_line(pts_in, mu_in, eig_axes_in, 2)
    if Nout :
        # first need to calc mean+axes of this new point-cloud
        mu_out, eig_axes_out, cov, U, S = lpc.calc_slope_int_pts(pts_out)
        pts_out0, pts_out_bnd0 = lpc.calc_bnds_along_line(pts_out, mu_out, eig_axes_out, 0)
        pts_out1, pts_out_bnd1 = lpc.calc_bnds_along_line(pts_out, mu_out, eig_axes_out, 1)
        pts_out2, pts_out_bnd2 = lpc.calc_bnds_along_line(pts_out, mu_out, eig_axes_out, 2)

    # ---------------- start figure -----------------------

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # setup the plot so a sphere kinda looks like a sphere
    fig.set_size_inches(8, 8)
    lims = lpc.calc_axes_plt_limits(pts)
    ax.axes.set_xlim3d(lims[0,0], lims[0,1]) 
    ax.axes.set_ylim3d(lims[1,0], lims[1,1]) 
    ax.axes.set_zlim3d(lims[2,0], lims[2,1]) 

    # plot axes of full point-cloud
    ax.plot3D(*pts0.T, c='C4', ls = '--')
    ax.plot3D(*pts1.T, c='C4', ls = '--')
    ax.plot3D(*pts2.T, c='C4', ls = '--', label='all pts')

    #ax.scatter(*pts.T, c='k', s=10*mdist, label='pts_all')
    
    # IN: lines and points
    if Nin :
        ax.plot3D(*pts_in0.T, c='C3')
        ax.plot3D(*pts_in1.T, c='C3')
        ax.plot3D(*pts_in2.T, c='C3')
        ax.scatter3D(*pts_in.T, c='C3', label='pts_in')

    #OUT: lines and points
    if Nout :
        ax.plot3D(*pts_out0.T, c='C0')
        ax.plot3D(*pts_out1.T, c='C0')
        ax.plot3D(*pts_out2.T, c='C0')
        ax.scatter3D(*pts_out.T, c='C0', label='pts_out')
    
    plt.ion()
    plt.legend()
    plt.show()

    #plt.savefig("fig_point-cloud_demo.svg")
    # ... because github doesn't support displaying SVG files???
    plt.savefig("fig_point-cloud_demo.png")

    # ---------------- print table report -----------------------

    otxt = "report_point-cloud_demo.txt"
    fff  = open(otxt, mode='w')    
    
    # for: all, non-outlier, outlier
    table_top = 'All points (N = {})'.format(N)
    table_pts = lpc.tableize_bounds(pts, mu, eig_axes,
                                    label=table_top)
    print(table_pts)
    fff.write(table_pts)

    if Nin :
        table_top_in = 'Inlier points (Nin = {})'.format(Nin)
        table_pts_in = lpc.tableize_bounds(pts_in, mu_in, eig_axes_in,
                                           label=table_top_in)
        print(table_pts_in)
        fff.write(table_pts_in)

    if Nout :
        table_top_out = 'Outlier points (Nout = {})'.format(Nout)
        table_pts_out = lpc.tableize_bounds(pts_out, mu_out, eig_axes_out,
                                            label=table_top_out)
        print(table_pts_out)
        fff.write(table_pts_out)

    fff.close()
    sys.exit(0)


