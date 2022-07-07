import sys, copy

import numpy as np

# -----------------------------------------------------------------------
# A small library of functions for visualizing and assessing some
# properties of point-clouds.
#
# Primarily, these calculate the mean and meaningful axes (using SVD)
# for the point-cloud.  There is also some simple outlier detection,
# using project and IQRs along each axis.  Finally, there is some
# estimation of the spread of the point-cloud, again along the
# meaningful axes of the cloud itself.
#
#
# ver = 1.0 (Nov 18, 2021)
# ver = 2.0 (Nov 22, 2021)
# ver = 3.0 (Jan  6, 2022)
# ver = 3.1 (Mar 22, 2022)  introduce Malahanobis dist, for outlierizing
# 
# written by PA Taylor (SSCC, NIMH, NIH)
# -----------------------------------------------------------------------

EPS = 1e-6
BIG = 10e10

# these 3-vectors are used as references for determining what are
# relatively "positive" and "negative" directions in different cases,
# purely for display; see positivize_slope(), below
vec111 = np.array([0.57735027, 0.57735027, 0.57735027])
vec001 = np.array([0, 0, 1])
vec100 = np.array([1, 0, 0])

# -----------------------------------------------------------------------
# -----------------------------------------------------------------------

def make_test_pts(N, D=3, frac_out=0.0, seed=5):
    """Make an example set of N points in D=3 dimensions (with a bit of
directional asymmetric, so it isn't just a sphere, for interest).

    Parameters
    ----------
    N         :(int) number of points to generate
    D         :(int) number of dimensions to work in;  much of this code
               is flexible for D values other than 3, but perhaps not all.
               So, in practice, leave this at 3 for now.
    frac_out  :(float) fraction of outliers to generate, so should be between
               [0, 1].
    seed      :(int) set the seed for the new, fancy 'best practice' way
               of setting a RNG seed (it's local).

    Return 
    ------
    C         :(array, 2D) array of size (N, D).

    """

    rng = np.random.default_rng(seed)

    A = 50+rng.random((N, D))
    B = 50+rng.random(N)
    C = np.zeros((N, D))
    for i in range(N):
        C[i] = A[i]/B[i]

    # number of outliers to generate
    Nout = np.min([np.max([0, int(frac_out*N)]), N])
    if Nout :
        print("++ Generating outliers: {} / {} points".format(Nout, N))
        C[0:Nout,:]*= 1.05    # such cute, little outliers!

    return C

# -----------------------------------------------------------------------

def demean_pts(pts):
    """Calculate the mean along each column of an array of points.  Return
both the mean vector, and the demeaned array of points.

    Parameters
    ----------
    pts     : (np.ndarray, 2D) typically an array of N columns (one per 
              point) and D dimensions (e.g., 3 for default applications here)

    Return
    ------
    mu      : (np.ndarray, 1D) array of column-wise means; has length D, the
              number of columns of pts (i.e., the space dimensionality)
    pts-mu  : (np.ndarray, 2D) array of same shape as input pts; the demeaned
              points
    """

    N, D = np.shape(pts)
    mu   = np.zeros(D)

    for i in range(D):
        mu[i] = np.mean(pts[:, i])
    
    return mu, pts-mu

def positivize_slope(slope):
    """Take an input slope that represents orientation, and determine
whether it should keep or flip its directionality, based on a small
tree of 'positive direction' considerations.

Define 'positive' to be having a dot product >0 with this diag vector
(1,1,1).  If orthogonal to that, then with (0,0,1); then with (1,0,0).

    Parameters
    ----------
    slope   : (np.ndarray, 1D) values of a slope in D=3 dimensions (so, 
              length = D).

    Return
    ------
    slope   : (np.ndarray, 1D) same slope as input, perhaps mult by -1

    """

    dot = np.dot(slope, vec111)
    if abs(dot) > EPS:
        if   dot > 0 :          return  slope
        elif dot < 0 :          return -slope

    # if that is *really* zero, then go by pointing upward
    dot = np.dot(slope, vec001)
    if abs(dot) > EPS:
        if   dot > 0 :          return  slope
        elif dot < 0 :          return -slope

    # if that is *really* zero, then go by pointing rightward
    dot = np.dot(slope, vec100)
    if abs(dot) > EPS:
        if   dot > 0 :          return  slope
        elif dot < 0 :          return -slope

    # and this point should never be reached
    print("Amazing vector:", slope)
    return slope


# NB: This is a primary function here.
def calc_slope_int_pts(pts):
    """For a given set of points pts, use SVD to calculate the pieces
    needed for a 'point-intercept' line formula in arbitrary dims.

    NB: the slope is an orientation, and could be equivalently
    multiplied by -1 with no harm done.  See positivize_slope() for
    the rules of how we assign convenient directionality.

    Parameters
    ----------
    pts     : (np.ndarray, 2D) typically an array of N columns (one per 
              point) and D dimensions (e.g., 3 for default applications here)

    Return
    ------
    mean    : (np.ndarray, 1D) array of column-wise means; has length D, the
              number of columns of pts (i.e., the space dimensionality)
    VH      : (np.ndarray, 2D) array of slopes, shape = (D, D), AKA the
              estimated orthonormal basis (axes) for pts from the
              eigenvectors, called 'eig_axes' in other functions;
              this is the 'vh' output of np.linalg.svd()
    COV     : (np.ndarray, 2D) covariance matrix of the points
    U       : (np.ndarray, 2D) array/matrix, shape = (N, N); prob not used;
              this is the 'u' output of np.linalg.svd()
    S       : (np.ndarray, 1D) array/matrix, len = N; prob not used;
              this is the 's' output of np.linalg.svd()

    """

    # first demean points
    mean, pts_dem = demean_pts(pts)

    # then do SVD:  VH will be orthonormal basis (slopes) for points
    U, S, VH      = np.linalg.svd(pts_dem)

    COV = np.cov(pts.T)

    for i in range(len(VH)):
        VH[i] = positivize_slope(VH[i])

    return mean, VH, COV, U, S

# -----------------------------------------------------------------------

def project_p_to_plane(p, mu, eig_axes, proj_idx):
    """Project point p onto the plane defined using the mean mu and two of
    the eig_axes axes.  The axis along which the point is projected
    is given by proj_idx.

    Parameters
    ----------
    p           : (array, 1D) a point in space, len=D
    mu          : (array, 1D) the mean of the point-cloud, len=D
    eig_axes    : (array, 2D) the three slopes/axes of the point-cloud,
                  shape = (D,D)
    proj_idx    : (int) the index of the particular slope/axis within
                  eig_axes *along which* we project; ergo, the point p ends
                  up on the plane defined by the other two slopes/axes

    Returns
    -------
    out         : (array, 1D) the new, post-projection point coords, len=D

    """

    D = len(mu)
    if proj_idx >= D :
        print("** ERROR: proj index must be in range [0, {}]".format(D))
        sys.exit(3)

    nvec = eig_axes[proj_idx]                 # slope along which we proj 
    fac  = np.dot(nvec, mu-p)

    return p + fac*nvec

def project_pts_to_plane(pts, mu, eig_axes, proj_idx):
    """See the help of project_p_to_plane() for all parameters; pts is
just an array of N 'p' values.

    Also, the output of this function is an NxD array of all new
    projected points.

    """

    N, D = np.shape(pts)
    out = np.zeros((N, D))

    for i in range(N):
        out[i] = project_p_to_plane(pts[i], mu, eig_axes, proj_idx)

    return out

# -----------------------------------------------------------------------

def project_p_to_line(p, mu, eig_axes, line_idx):
    """Perform two consecutive planar projections on a point p, so that a
3D point is projected onto a line/eigenvector of the point-cloud
(given by mu and eig_axes).  See the help of project_p_to_plane()
for all parameters; *except* in this case line_idx is the (int) index
of the slope/axis *onto which the point is projected.*

    Return
    ------
    q         : (array, 1D) the new, post-projection point coords, len=D

    qdist     : (float) the signed distance from the mean, given by
                dotprod with projected eigenvector; hence, this 'distance'
                value can be positive or negative.

    """

    D = len(mu)
    if line_idx >= D :
        print("** ERROR: line index must be in range [0, {}]".format(D))
        sys.exit(3)

    all_idx = [0, 1, 2]
    all_idx.remove(line_idx)

    # define q which gets projected twice.  order of proj doesn't matter
    q = copy.deepcopy(p)
    for idx in all_idx :
        q = project_p_to_plane(q, mu, eig_axes, idx)

    qdist = np.dot(eig_axes[line_idx], q-mu)

    return q, qdist

def project_pts_to_line(pts, mu, eig_axes, line_idx):
    """Perform two consecutive planar projections on each of the N points
in the pts array, so that each 3D point is projected onto a
line/eigenvector of the point-cloud (given by mu and eig_axes).

    Parameters
    -----------
    pts           : (arr, 2D) the N points, shape = (N, D)
    mu            : (arr, 1D) the coord mean of the points, len=D
    eig_axes      : (arr, 2D) array of 3 slopes/axes from SVD, shape = (D, D)
    line_idx      : (scalar) index in range [0, D-1], for the axis onto which
                    the projection is done
    
    Return
    ------
    pts_proj      : (3xN arr) the points projected along the line
    pts_proj_dist : (arr, len=N) the scalar 'coordinate' of each point along
                    the line
    """

    N, D          = np.shape(pts)
    pts_proj      = np.zeros((N, D))
    pts_proj_dist = np.zeros(N)

    # project each point
    for i in range(N):
        pts_proj[i], pts_proj_dist[i] = project_p_to_line(pts[i], mu, 
                                                          eig_axes, 
                                                          line_idx)

    return pts_proj, pts_proj_dist

def calc_mahalanobis_dist(pts, mu, cov):
    """Do now Malahanobis packing.

    """

    N, dim = np.shape(pts)

    Cinv = np.linalg.inv(cov)   # inv of cov mat
    Pdem = pts - mu             # demeaned points

    mdist = np.zeros(N)
    for i in range(N):
        mdist[i] = np.matmul(Pdem[i].T, np.matmul(Cinv, Pdem[i]))
        if mdist[i] >= 0 :
            mdist[i] = mdist[i]**0.5
        else:
            print("WARN: Mahalanobis distance will be imaginary for\n"
                  "      point [{}].  Setting it to 0.".format(i))
            mdist[i] = 0

    return mdist



# NB: This is a primary function here.
def calc_bnds_along_line(pts, mu, eig_axes, line_idx):
    """For the N pts that have mean mu and axes eig_axes, calculate the
boundary coordinates of the point cloud along one slope/axis, and
then also the distance of pair of boundaries along that line.

    Parameters
    -----------
    pts          : (arr, 2D) the N points, shape = (N, D)
    mu           : (arr, 1D) the coord mean of the points, len=D
    eig_axes     : (arr, 2D) array of 3 slopes/axes from SVD, shape = (D, D)
    line_idx     : (scalar) index in range [0, D-1], for the axis onto which
                   the projection is done

    Return
    ------
    bnds         : (array, 2D) the min/max boundary points along the line,
                   shape = (D, 3)
    bnds_dist    : (arr, len=2) the scalar 'coordinate' of each boundary point 
                   along the line; that is, the distance of the boundary along
                   each direction of the slope/line
    """

    N, D = np.shape(pts)

    pts_proj, pts_proj_dist = project_pts_to_line(pts, mu, 
                                                  eig_axes, line_idx)

    minp =  BIG
    maxp = -BIG
    for i in range(N):
        if pts_proj_dist[i] < minp:
            minp      = pts_proj_dist[i]
            minp_coor = pts_proj[i]
        if pts_proj_dist[i] > maxp:
            maxp      = pts_proj_dist[i]
            maxp_coor = pts_proj[i]

    bnds      = np.array([minp_coor, maxp_coor])
    bnds_dist = np.array([minp, maxp])

    return bnds, bnds_dist

# -----------------------------------------------------------------------

def box_iqr_arr(arr, perc=(25, 75), iqr_fac=1.5):
    """For a 1D array arr of len=N (which might represent signed distance
values along a line/axis onto which a point-cloud of N points was
projected), find the 'interquartile range' (IQR) given by the perc
limits (we call this IQR because of the defaults; a user could change
them, and it would then just be inter[some] range).  

From this IQR, calculate the 'non-outlier' interval (non_out), which
is the IQR boundaries plus/minus iqr_fac*IQR.  Also calculate the
True/False index array 'filt_non_out', locating non-outliers/outliers,
respectively.

    Return
    ------
    iqr          :(float) the size of the IQR
    non_out      :(array, 1D) the boundaries of the interval within which
                  points are *not* outliers (beyond this, thar be dragons),
                  len=2
    filt_non_out :(array, 1D) an array of True/False values, where True
                  values represent locations whose points are *not* outliers,
                  and False, those that are; len=N

    """

    if perc[1] < perc[0] :
        print("** ERROR: Cannot have upper IQR perc bound '{}' "
              "< lower IQR per bound '{}'"
              "".format(perc[1], perc[0]))
        sys.exit(3)

    iqr     = np.percentile(arr, perc)
    iqr_mag = iqr[1] - iqr[0]
    non_out = np.array([iqr[0] - iqr_fac*iqr_mag, iqr[1] + iqr_fac*iqr_mag])

    filt_non_out = np.array(arr > non_out[0])
    filt_non_out*= np.array(arr < non_out[1])

    return iqr, non_out, filt_non_out


# NB: This is a primary function here.
def filter_no_iqr_out(pts, mu, eig_axes, perc=(25, 75), iqr_fac=1.5):
    """For a given set of point-cloud pts that have mean mu and axes
    eig_axes, determine outliers (based on simple projection along
    each cloud axis, not to be confused with the novel).  Provide the
    filter that shows which points are *not* outliers, and then
    provide separate arrays of non-outlier points and outlier-only
    points.

    The filter process is:
    0) determine the perc[0]-to-perc[1] %ile range of points along that axis
    1) define the IQR as the interval of those percentiles
    2) determine outliers as being outside the IQR plus/minus iqr_fac * IQR.
    3) make a True/False index array 'filt_non_out', locating 
       non-outliers/outliers, respectively.
    4) return filt_non_out arr, the arr of filtered points, and the arr of 
       outliers

    Parameters
    ----------
    pts          :(np.ndarray, 2D) typically an array of N rows (one per 
                  point) and D columns (the number of dimensions, typically
                  3 here)
    mu           :(array, 1D) the mean of the point-cloud, len=D
    eig_axes     :(array, 2D) the three slopes/axes of the point-cloud,
                  shape = (D,D)
    perc         :(tuple, 1D) the pair of percentile values used to make the 
                  IQR with default being (25, 75);  each element should be
                  in range [0, 100], and changing these values means you 
                  aren't really using 'quartiles' any more.
    iqr_fac      :(float) the scale factor used to calculate what an outlier
                  is from the IQR: if the IQR spans values A and B, so that
                  the IQR span is D=B-A, then non-outiers are point within:
                  [A-D*iqr_fac, B+D*iqr_fac]

    Return
    ------
    filt_non_out :(array, 1D) an array of True/False values, where True
                  values represent locations whose points are *not* outliers,
                  and False, those that are; len=N
    pts_non_out  :(array, 2D) an array of *non-outlier* points, which will 
                  have D columns, and the number of rows will be <=N,
                  depending on the number of outliers
    pts_only_out :(array, 2D) an array of *outlier* points, which will have
                  D columns, and the number of rows will be <=N, depending
                  on the number of outliers

    """

    # get projected points, and projected points distances
    pts_p0, pts_pd0 = project_pts_to_line(pts, mu, eig_axes, 0)
    pts_p1, pts_pd1 = project_pts_to_line(pts, mu, eig_axes, 1)
    pts_p2, pts_pd2 = project_pts_to_line(pts, mu, eig_axes, 2)

    # use IQRs of each projected set of values to determine outliers
    iqr0, non_out0, filt0 = box_iqr_arr(pts_pd0, perc=perc, iqr_fac=iqr_fac)
    iqr1, non_out1, filt1 = box_iqr_arr(pts_pd1, perc=perc, iqr_fac=iqr_fac)
    iqr2, non_out2, filt2 = box_iqr_arr(pts_pd2, perc=perc, iqr_fac=iqr_fac)

    # merge non-outlier filters (and invert to get outlier filter)
    filt_non_out  = filt0 * filt1 * filt2
    filt_only_out = (1 - filt_non_out).astype(bool)

    return filt_non_out, pts[filt_non_out], pts[filt_only_out]

# -----------------------------------------------------------------------

# NB: This is a primary function here (sigh).
def calc_axes_plt_limits(pts):
    """To help plot in 3D in Python, one might want equal ranges for the
    x-, y-, and z-axis limits, centered around the mean of the
    point-cloud pts.  This function finds the minimal range of
    symmetric, equal magnitude ranges for the axes that will span all
    pts.

    Parameters
    ----------
    pts     : (np.ndarray, 2D) typically an array of N columns (one per 
              point) and D dimensions (e.g., 3 for default applications here)

    Return
    ------
    lims    :(np.ndarray, 2D) an array of pairs of limits for each axis
             among D dimensions, so shape = (D, 2).

    """

    N, D   = np.shape(pts)
    tmp    = np.zeros(D)
    lims   = np.zeros((D, 2))

    # get mean of cloud, and demeaned point values (because we want to
    # use distance from mean)
    mu, pts_dem = demean_pts(pts)

    # build a list of max (unsigned) dist from mu along each axis
    for j in range(D):
        tmp[j]  = np.max(np.abs(pts_dem[:,j]))
        #tmp[j]  = np.max(np.abs(pts[:,j] - mu[j]))

    # pick the max of the above to be the window size
    win = np.max(tmp)

    for j in range(D):
        lims[j] = (mu[j]-win, mu[j]+win)

    return lims

# -----------------------------------------------------------------------

# NB: This is a primary function here.
def tableize_bounds(pts, mu, eig_axes, label=''):
    """Make a report on the bounds of the input values.

    Parameters
    ----------
    pts          : (arr, 2D) the N points, shape = (N, D)
    mu           : (arr, 1D) the coord mean of the points, len=D
    eig_axes     : (arr, 2D) array of 3 slopes/axes from SVD, shape = (D, D)
    label        : (str) text to put at the top of the table

    Return
    ------
    ostr         : (str) The lovely table.

    """

    D = len(mu)

    # calc the span values to report along each axis
    bnds_dist = np.zeros((D, 2))
    for ii in range(D):
        tmp1, bnds_dist[ii,:] = calc_bnds_along_line(pts, mu, eig_axes, ii)

    # start building the table, as a sum of strings
    ostr = '''\n{}\n'''.format( '='*80 )

    if label :
        ostr+= '''{}\n\n'''.format( label )

    ostr+= '''The mean coordinate of the points is at: '''
    ostr+= '''({:0.4f}, {:0.4f}, {:0.4f})\n\n'''.format( mu[0], mu[1], mu[2] )

    ostr+= '''The axes are oriented as follows, '''
    ostr+= '''with min and max distances from the mean point:\n'''

    for ii in range(D):
        aa = ', '.join(['{:7.4f}'.format(x) for x in eig_axes[ii,:]])
        aa = 'Axis {}: ('.format(ii) + aa + '), '
        bb = ', '.join(['{:7.4f}'.format(x) for x in bnds_dist[ii,:]])
        bb = 'span: [' + bb + '], '
        cc = '{:7.4f}'.format(abs(bnds_dist[ii,1]-bnds_dist[ii,0]))
        cc = 'interval: ' + cc + '\n'

        ostr+= aa + bb + cc

    ostr+= '''\n{}'''.format( '='*80 )

    return ostr

# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
