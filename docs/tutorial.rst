=========
Tutorial
=========
synthesizAR provides a multi-stage pipeline for generating synthetic observations of the solar corona. This tutorial provides a basic outline of the steps in that pipeline. More details about each step can be found in the relevant API documentation.

Computing a Synthetic Magnetogram
---------------------------------
We will first need to create a synthetic magnetogram to perform a field extrapolation from. `synthesizAR` provides some basic functionality for creating synthetic magnetograms. First, import the needed libraries ::

    >>> import os
    >>> import astropy.units as u
    >>> import astropy.constants as const
    >>> from astropy.coordinates import SkyCoord
    >>> from sunpy.coordinates import HeliographicStonyhurst, Helioprojective
    >>> import synthesizAR
    >>> import synthesizAR.extrapolate
    >>> import yt; yt.funcs.mylog.setLevel(50); yt.config.ytcfg.set('yt', 'suppressStreamLogging','True') # This is only to suppress unneeded output
    >>> import tempfile
    >>> tmpdir = tempfile.mkdtemp() # All example output is redirected to a temp folder. Change this to whatever directory you want your results in

Then, configure our magnetogram. We need to specify the bounds of our map, number of "sunspots", the spread of these spots, and their strength, ::

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

    >>> coordinates, field_strengths = extrapolator.trace_fieldlines(B_field, 100, verbose=False, notebook=False)

Each entry in our fieldline list contains 1) the coordinates of the fieldline as a `~astropy.coordinates.SkyCoord` object and 2) the magnitude of the magnetic field strength. We can then use these fieldlines and the associated magnetogram to construct our `~synthesizAR.Skeleton` object,

    >>> active_region = synthesizAR.Skeleton.from_coordinates(coordinates, field_strengths)

Note that we need not construct the fieldlines in this manner. Similary, we could get the fieldlines using some other method (e.g. the `pfss package from SSW <http://www.lmsal.com/~derosa/pfsspack/>`_) and construct our active region in exactly the same way.

Loop Thermal Structure
------------------------
The next step is to calculate the temperature and density as a function of loop coordinate and time. For this tutorial, we will compute a hydrostatic solution using the scaling laws of `Martens (2010) <http://adsabs.harvard.edu/abs/2010ApJ...714.1290M>`_.

In order to compute the thermal structure along each loop, we need an interface between the `~synthesizAR.Loop` object and the model being used. synthesizAR provides interfaces to several different models. You can also easily define your own. Let's create the interface and compute the thermal structure for all of our loops,

    >>> from synthesizAR.interfaces import MartensInterface
    >>> martens = MartensInterface()
    >>> active_region.load_loop_simulations(martens, os.path.join(tmpdir, 'loops.h5'), notebook=False)

Synthesizing AIA Intensity Maps
-------------------------------
The final step in our forward modeling pipeline is to compute the synthetic intensity as observed by some instrument. In this case, we'll forward model intensities as observed by AIA, but you can also define your own instrument.

First, we need to create the instrument and tell it the location of our observer. In this case, we'll use the observer coordinate define by our magnetogram: at disk center at a distance of 1 AU.

    >>> from synthesizAR.instruments import InstrumentSDOAIA
    >>> aia = InstrumentSDOAIA([0,1]*u.s, magnetogram.observer_coordinate)

Next, we combine our instrument object and active region into a single observer object that will compute the intensities from the temperatures and densities of each loop.

    >>> observer = synthesizAR.Observer(active_region, [aia])
    >>> observer.build_detector_files(tmpdir, 1*u.Mm)

In order to compute the intensities in each pixel along the line of sight, we first compute and interpolate the intensities over all loop lengths and simulation times. Then, we bin these counts over the whole pixel array.

    >>> observer.flatten_detector_counts()
    >>> observer.bin_detector_counts(tmpdir)

This last step produces a `~sunpy.map.GenericMap` for each detector channel at each timestep and saves it to a FITS file. For example, to load the map for the 94 :math:`\mathrm{\mathring{A}}` channel

    >>> from sunpy.map import Map
    >>> m = Map(os.path.join(tmpdir, 'SDO_AIA', '94', 'map_t000000.fits'))

Note that there is only one map per channel at :math:`t=0` s because the thermal structure of the loops in the active region is determined by the hydrostatic scaling laws.
