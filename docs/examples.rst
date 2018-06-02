============
Examples
============

Below are some example scripts for accomplishing tasks with or related to the synthesizAR package.

Downloading and Prepping an HMI Magnetogram
--------------------------------------------
To build an active region, we need to start with a line-of-sight magnetogram. Magnetogram data is typically packaged in FITS files and are available from a variety of sources, including `JSOC <http://jsoc.stanford.edu/>`_ or `Helioviewer <https://helioviewer.org/>`_. Using SunPy, we can also query the VSO and fetch the data using Python. We'll grab an HMI magnetogram from 1 January 2013. Now that we have our magnetogram, we need to manipulate the map in a few different ways. First, we'll rotate it so that the physical and pixel coordinates are oriented in the same direction. Next, we'll crop the image to the area on the disk we are interested in. We are interested in active region NOAA 11640. Finally, we resample the cropped magnetogram to 100-by-100 pixels.

.. plot::
    :include-source:

    import os
    import tempfile
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from sunpy.net import Fido,attrs as a,vso
    from sunpy.time import TimeRange
    from sunpy.map import Map
    # Download data
    q = Fido.search(a.Time(TimeRange('2013/01/01 00:03:30', '2013/01/01 00:04:30')),
                    a.Instrument('HMI'), vso.attrs.Physobs('LOS_magnetic_field'))
    with tempfile.TemporaryDirectory() as tmpdirname:
        file_path = Fido.fetch(q, path=os.path.join(tmpdirname, '{file}.fits'))
        magnetogram = Map(file_path)
    # Rotate
    magnetogram = magnetogram.rotate(order=3)
    # Crop to AR
    blc = SkyCoord(-250*u.arcsec, 350*u.arcsec, frame=magnetogram.coordinate_frame)
    trc = SkyCoord(75*u.arcsec, 650*u.arcsec, frame=magnetogram.coordinate_frame)
    magnetogram_ar = magnetogram.submap(blc,trc)
    # Resample
    magnetogram_ar_resampled = magnetogram_ar.resample([100, 100]*u.pixel)
    # Plot
    fig = plt.figure(figsize=(20,8))
    ax = fig.add_subplot(131,projection=magnetogram)
    magnetogram.plot(title='Full',axes=ax,cmap='hmimag',vmin=-1.5e3,vmax=1.5e3)
    magnetogram.draw_rectangle(blc,trc.Tx - blc.Tx, trc.Ty - blc.Ty, axes=ax, color='k', lw=2)
    ax = fig.add_subplot(132,projection=magnetogram_ar)
    magnetogram_ar.plot(title='Active Region',axes=ax,cmap='hmimag',vmin=-1.5e3,vmax=1.5e3)
    ax = fig.add_subplot(133,projection=magnetogram_ar_resampled)
    magnetogram_ar_resampled.plot(title='Resampled',axes=ax,cmap='hmimag',vmin=-1.5e3,vmax=1.5e3)
    plt.tight_layout()
    plt.show()

