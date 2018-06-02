=========
Tutorial
=========
synthesizAR provides a multi-stage pipeline for generating synthetic observations of the solar corona. This tutorial provides a basic outline of the steps in that pipeline.

Computing a Synthetic Magnetogram
---------------------------------
We will first need to create a synthetic magnetogram to perform a field extrapolation from. `synthesizAR` provides some basic
functionality for creating synthetic magnetograms. First, import the needed libraries ::

    >>> import astropy.units as u
    >>> import astropy.constants as const
    >>> from astropy.coordinates import SkyCoord
    >>> from sunpy.coordinates import HeliographicStonyhurst, Helioprojective
    >>> import synthesizAR.extrapolate
    >>> import yt; yt.funcs.mylog.setLevel(50); yt.config.ytcfg.set('yt', 'suppressStreamLogging','True') # This is only to suppress unneeded output

Then, configure our magnetogram. We need to specify the bounds of our map, number of "sunspots", the spread of these spots,
and their strength, ::

    >>> shape = [ 50, 50] * u.pixel
    >>> obs = SkyCoord(lon=0.*u.deg,lat=0.*u.deg,radius=const.au,frame=HeliographicStonyhurst)
    >>> blc = SkyCoord(-50 * u.arcsec, -50 * u.arcsec,frame=Helioprojective(observer=obs))
    >>> trc = SkyCoord(50 * u.arcsec, 50 * u.arcsec, frame=Helioprojective(observer=obs))
    >>> centers = SkyCoord([15, -15,]*u.arcsec,[0, 0,]*u.arcsec,frame=Helioprojective(observer=obs))
    >>> sigmas = u.Quantity([[5, 5], [5, 5],],'arcsec')
    >>> amplitudes = u.Quantity([1e3, -1e3,], 'Gauss')

Finally, create the synthetic magnetogram. This is just a normal `~sunpy.map.GenericMap` object, ::

    >>> magnetogram = synthesizAR.extrapolate.synthetic_magnetogram(blc, trc, shape, centers, sigmas, amplitudes, observer=obs)

Extrapolating a Magnetic Field 
------------------------------
Now that we have a magnetogram, we can use it to perform a potential field extrapolation to 
approximate the 3D vector magnetic field. We first need to specify the shape and spatial extent of 
the z-dimension and we set the spatial extent of the z-dimension such that the resolution is the same 
as in the x-dimension, ::

    >>> shape_z = 50 * u.pixel
    >>> width_z = (magnetogram.scale.axis1 * shape_z).to(u.radian).value * magnetogram.dsun

Now we can setup the extrapolator and compute :math:`\vec{B}`,

    >>> extrapolator = synthesizAR.extrapolate.PotentialField(magnetogram, width_z, shape_z)
    >>> B_field = extrapolator.extrapolate()

Building an Active Region and Tracing Fieldlines
------------------------------------------------
Now that we have the 3D magnetic field, we can trace magnetic fieldlines through the volume,

    >>> fieldlines = extrapolator.trace_fieldlines(B_field, 100, verbose=False, notebook=False)

Each entry in our fieldline list contains 1) the coordinates of the fieldline as a `~astropy.coordinates.SkyCoord` object and 2) the magnitude of the magnetic field strength. We can then use these fieldlines and the associated magnetogram to construct our `~synthesizAR.Field` object,

    >>> active_region = synthesizAR.Field(magnetogram, fieldlines)

Note that we need not construct the fieldlines in this manner. Similary, we could get the fieldlines using some other method (e.g. the `pfss package from SSW <http://www.lmsal.com/~derosa/pfsspack/>`_) and construct our active region in exactly the same way.

Building a Hydrodynamic Interface
---------------------------------

Synthesizing AIA Intensity Maps
-------------------------------
