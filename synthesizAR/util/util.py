"""
Some basic tools/utilities needed for active region construction. These functions are generally
peripheral to the actual physics.
"""

import itertools

import numpy as np
import numba
import astropy.units as u
import solarbextrapolation.utilities

__all__=['convert_angle_to_length','find_seed_points','collect_points',
         '_numba_lagrange_interpolator','_numba_interpolator_wrapper']

def convert_angle_to_length(hmi_map,angle_or_length,working_units=u.meter):
    """
    Helper for easily converting between angle and length units. If converting to length, returned units will be `~astropy.units.cm`. If converting to angle, the returned units will be `~astropy.units.arcsec`.
    """
    observed_distance = (hmi_map.dsun - hmi_map.rsun_meters)
    radian_length = [(u.radian,u.meter,lambda x: observed_distance*x, lambda x: x/observed_distance)]
    converted = solarbextrapolation.utilities.decompose_ang_len(angle_or_length, working_units=working_units, equivalencies=radian_length)

    if working_units == u.meter:
        return converted.to(u.cm)
    else:
        return converted.to(u.arcsec)

def find_seed_points(volume, boundary_map, number_fieldlines, preexisting_seeds=[], mask_threshold=0.05, safety=1.2, max_failures=1000):
    """
    Given a 3D extrapolated field and the corresponding magnetogram, estimate the locations of the seed points for the fieldline tracing through the extrapolated 3D volume.

    Parameters
    ----------
    volume : `~yt.frontends.stream.data_structures.StreamDataset`
        Dataset containing the 3D extrapolated vector field
    boundary_map : `~sunpy.map.Map`
        HMI magnetogram
    number_fieldlines : `int`
        Number of seed points
    """
    #mask the boundary map and estimate resampled resolution
    mask_above = mask_threshold*boundary_map.min()
    masked_boundary_map = np.ma.masked_greater(boundary_map.data,mask_above)
    epsilon_area = float(masked_boundary_map.count())/float(boundary_map.data.shape[0]*boundary_map.data.shape[1])
    resample_resolution = int(safety*np.sqrt(number_fieldlines/epsilon_area))

    #resample and mask the boundary map
    boundary_map_resampled = boundary_map.resample([resample_resolution,resample_resolution]*(boundary_map.xrange/boundary_map.scale.x).unit, method='linear')
    masked_boundary_map_resampled = np.ma.masked_greater(boundary_map_resampled.data,mask_above)

    #find the unmasked indices
    unmasked_indices = [(ix,iy) for iy,ix in zip(*np.where(masked_boundary_map_resampled.mask==False))]

    if len(unmasked_indices) < number_fieldlines:
        raise ValueError('Requested number of seed points too large. Increase safety factor.')

    length_x = convert_angle_to_length(boundary_map_resampled,boundary_map_resampled.xrange)
    length_y = convert_angle_to_length(boundary_map_resampled,boundary_map_resampled.yrange)
    x_pos = np.linspace(length_x[0].value, length_x[1].value, resample_resolution)
    y_pos = np.linspace(length_y[0].value, length_y[1].value, resample_resolution)

    #choose seed points
    seed_points = []
    i_fail = 0
    z_pos = volume.domain_left_edge.value[2]
    while len(seed_points) < number_fieldlines and i_fail<max_failures:
        choice = np.random.randint(0,len(unmasked_indices))
        ix,iy = unmasked_indices[choice]
        _tmp = [x_pos[ix],y_pos[iy],z_pos]
        if _tmp not in preexisting_seeds:
            seed_points.append(_tmp)
            i_fail = 0
        else:
            i_fail += 1
        del unmasked_indices[choice]

    if i_fail == max_failures:
        raise ValueError('Could not find desired number of seed points within failure tolerance of {}. Try increasing safety factor or the mask threshold'.format(max_failures))

    return seed_points

def collect_points(x,y):
    """
    Using two lists, where the first has repeated entries, sum the corresponding entries
    in the repeated list for each unique entry in the first list.

    Parameters
    ----------
    x : `list`
    y : `list`
    """
    unique_sorted_x = np.array(sorted(set(x)))
    summed_sorted_y = np.array([np.array([g[1] for g in grp]).sum(axis=0) \
            for lvl,grp in itertools.groupby(sorted(zip(x,y),key=lambda k:k[0]),lambda k:k[0])])
    return unique_sorted_x,summed_sorted_y

@numba.jit(nopython=True)
def _numba_lagrange_interpolator(x_data,y_data,x):
    #checks
    if len(x_data) != len(y_data):
        raise ValueError('x_data and y_data must be of equal length')
    if len(x_data) < 3:
        raise ValueError('Data must be at least length 3')
    if x>np.max(x_data) or x<np.min(x_data):
        raise ValueError('x is outside of the interpolation range.')

    # get the points
    if len(np.where(x==x_data)[0])>0:
        return y_data[np.where(x==x_data)[0][0]]
    else:
        i_min = np.where(x_data<x)[0][-1]
        i_max = np.where(x_data>x)[0][0]
        if i_min == 0:
            i_mid,i_max = 1,2
        else:
            i_min,i_mid = i_max-2,i_max-1

    x0,x1,x2 = x_data[i_min],x_data[i_mid],x_data[i_max]
    y0,y1,y2 = y_data[i_min],y_data[i_mid],y_data[i_max]

    # calculate Lagrange interpolation points
    term1 = (x - x1)*(x - x2)/(x0 - x1)/(x0 - x2)*y0
    term2 = (x - x0)*(x - x2)/(x1 - x0)/(x1 - x2)*y1
    term3 = (x - x0)*(x - x1)/(x2 - x0)/(x2 - x1)*y2

    return term1 + term2 + term3

@numba.jit(nopython=True)
def _numba_interpolator_wrapper(x_data,y_array_data,x,normalize=False,cutoff=0.):
    y = np.zeros(y_array_data.shape[1])
    for i in range(y_array_data.shape[1]):
        y[i] = _numba_lagrange_interpolator(x_data,y_array_data[:,i],x)
    y = np.where(y<cutoff,np.zeros(len(y)),y)
    if normalize:
        y /= np.sum(y)
    return y
