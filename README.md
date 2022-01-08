# proj_point-cloud

Some (very) simple functionality for visualizing and assessing the
spread of a cloud of points, in Python.

------------------------

Try running the `demo*.py` script.  The only library dependency for
`lib*.py` is NumPy.  The demo includes the matplotlib and mpl_toolkits
library, for visualization.

Running the `demo*.py` script should produce an image like this:
![fig_point-cloud_demo](https://user-images.githubusercontent.com/9322171/148624878-e7195257-e796-46c5-9d48-8de0db9ab108.png)
\.\.\. where the initial point-cloud contains all the points, of which
the blue ones were determine to be outliers (and the red ones are
therefore the inliers).  The red and blue lines show the calculated
axes of inliers and outliers, respectively (using SVD to generate a
meaningful, orthogonal basis for each set of points).  The dashed
purple lines are the axes for the original, full point-cloud.

There is also a text report that is generated, like this one:
[report_point-cloud_demo.txt](https://github.com/mrneont/proj_point-cloud/files/7832066/report_point-cloud_demo.txt)
It gives information on both the original/full point-cloud, as well as
each of the inlier and outlier ones.  In particular, the spread of
each point cloud along each of its new axes is noted.  One could also
easily calculate the standard deviation of the point-cloud along each
axis. 
