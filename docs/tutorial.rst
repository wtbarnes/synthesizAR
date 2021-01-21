=========
Tutorial
=========
synthesizAR provides a multi-stage pipeline for generating synthetic observations of the solar corona. This tutorial provides a basic outline of the steps in that pipeline. More details about each step can be found in the relevant API documentation.

Computing a Synthetic Magnetogram
---------------------------------
We will first need to create a synthetic magnetogram to perform a field extrapolation from. `synthesizAR` provides some basic functionality for creating synthetic magnetograms. First, import the needed libraries ::

    >>> import os
    >>> import distributed
    >>> import astropy.time
    >>> import astropy.units as u
    >>> import astropy.constants as const
    >>> from astropy.coordinates import SkyCoord
    >>> from sunpy.coordinates import HeliographicStonyhurst, Helioprojective
    >>> from sunpy.coordinates.ephemeris import get_earth
    >>> import synthesizAR
    >>> import synthesizAR.extrapolate
    >>> import yt; yt.funcs.mylog.setLevel(50); yt.config.ytcfg.set('yt', 'suppressStreamLogging','True') # This is only to suppress unneeded output
    >>> import tempfile
    >>> tmpdir = tempfile.mkdtemp() # All example output is redirected to a temp folder. Change this to whatever directory you want your results in

Then, configure our magnetogram. We need to specify the bounds of our map, number of "sunspots", the spread of these spots, and their strength, ::

    >>> shape = [ 50, 50] * u.pixel
    >>> obs = get_earth(astropy.time.Time.now())
    >>> hpc_frame = Helioprojective(observer=obs, obstime=obs.obstime)
    >>> blc = SkyCoord(-50 * u.arcsec, -50 * u.arcsec,frame=hpc_frame)
    >>> trc = SkyCoord(50 * u.arcsec, 50 * u.arcsec, frame=hpc_frame)
    >>> centers = SkyCoord([15, -15,]*u.arcsec,[0, 0,]*u.arcsec,frame=hpc_frame)
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

Before proceeding, we want to ensure that each of the loops in our active region is sufficiently sampled such that when we project along our LOS, our image is not "patchy." To ensure this, we can interpolate each of the loops in our active region to sufficient spatial resolution. What "sufficient" means here depends on how detailed we want our final image to be. As a rule of thumb, we want the grid cell widths of our loops to be less than a pixel of our observing instrument.

>>> ds = (0.6*u.arcsec).to('radian').value * const.au * 0.9
>>> active_region = active_region.refine_loops(ds)

Loop Thermal Structure
------------------------
The next step is to calculate the temperature and density as a function of loop coordinate and time. For this tutorial, we will compute a hydrostatic solution using the scaling laws of `Martens (2010) <http://adsabs.harvard.edu/abs/2010ApJ...714.1290M>`_.

Next, we need to set up our `~distributed.Client` instance which will handle the underlying parallelism.

    >>> client = distributed.Client(local_dir=tmpdir)

In order to compute the thermal structure along each loop, we need an interface between the `~synthesizAR.Loop` object and the model being used. synthesizAR provides interfaces to several different models. You can also easily define your own. Let's create the interface and compute the thermal structure for all of our loops,

    >>> from synthesizAR.interfaces import MartensInterface
    >>> martens = MartensInterface(1e-5*u.erg/(u.cm**3)/u.s)
    >>> status = active_region.load_loop_simulations(martens, os.path.join(tmpdir, 'loops.zarr'))

Synthesizing AIA Intensity Maps
-------------------------------
The final step in our forward modeling pipeline is to compute the synthetic intensity as observed by some instrument. In this case, we'll forward model intensities as observed by AIA, but you can also define your own instrument.

First, we need to create the instrument and tell it the location of our observer. In this case, we'll use the observer coordinate define by our magnetogram: at disk center at a distance of 1 AU.

    >>> from synthesizAR.instruments import InstrumentSDOAIA
    >>> aia = InstrumentSDOAIA([0,1]*u.s, magnetogram.observer_coordinate, pad_fov=(5,5)*u.arcsec)  # doctest: +REMOTE_DATA

Note that we are only observing at :math:`t=0` s as our loop model is a static model and thus our
forward-modeled intensities will not evolve in time.

Lastly, we "observe" our active region skeleton, combined with our hydrostatic loop simulations, and project them along
the line of sight as defined by our observer,

    >>> aia.observe(active_region, tmpdir, channels=aia.channels[2:3])

This produces a `~sunpy.map.GenericMap` at each timestep for the 171 :math:`\mathrm{\mathring{A}}` channel and
saves it to a FITS file. To load the resulting map,

    >>> from sunpy.map import Map
    >>> m = Map(os.path.join(tmpdir, 'm_171_t0.fits'))

Note that there is only one map per channel at :math:`t=0` s because the thermal structure of the loops
in the active region is determined by the hydrostatic scaling laws.
