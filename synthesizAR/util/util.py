"""
Some basic tools/utilities needed for active region construction. These functions are generally
peripheral to the actual physics.
"""

import itertools

import numpy as np
import astropy.units as u
import solarbextrapolation.utilities

__all__=['convert_angle_to_length','find_seed_points','collect_points']

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
    volume : yt uniform grid
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
    return np.array([np.array([g[1] for g in grp]).sum(axis=0) \
        for lvl,grp in itertools.groupby(sorted(zip(x,y),key=lambda x:x[0]),lambda x:x[0])])
