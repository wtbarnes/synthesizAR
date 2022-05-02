"""
Simulating AIA 171 Emission from a Bundle of Strands
=====================================================

This example shows how to model AIA emission from a
bundle of semi-circular strands for two different
viewpoints.
"""
import matplotlib.pyplot as plt
import astropy.time
import astropy.units as u
from astropy.visualization import quantity_support
from astropy.coordinates import SkyCoord
from sunpy.map import all_coordinates_from_map

import synthesizAR
from synthesizAR.instruments import InstrumentSDOAIA
from synthesizAR.interfaces import RTVInterface
from synthesizAR.models import semi_circular_bundle
# sphinx_gallery_thumbnail_number = -1

###########################################################################
# First, we calculate a list of coordinates comprising a semi-circular
# bundle of strands. The bundle has a length of 50 Mm and a cross-sectional
# radius of 1 Mm and a total of 500 strands. The strands are uniformly
# distributed over the bundle.
obstime = astropy.time.Time.now()
pos = SkyCoord(
    lon=0*u.deg, lat=0*u.deg, radius=1*u.AU, obstime=obstime, frame='heliographic_stonyhurst')
bundle_coords = semi_circular_bundle(50 * u.Mm, 1*u.Mm, 500, observer=pos)

###########################################################################
# As in other examples, we then use the coordinates of our strands to
# construct the `~synthesizAR.Skeleton` object.
strands = [synthesizAR.Loop(f'strand{i}', c) for i, c in enumerate(bundle_coords)]
bundle = synthesizAR.Skeleton(strands)
bundle.peek(observer=pos)

###########################################################################
# We can also look at our bundle of strands from the side to confirm it has
# the desired geometry.
side_on_view = SkyCoord(lon=0*u.deg, lat=-90*u.deg, radius=1*u.AU, frame=pos.frame)
bundle.peek(observer=side_on_view, grid_kwargs={'grid_spacing': 2*u.deg})

############################################################################
# We will again use a simple RTV loop model to compute the thermal structure
# of each strand. This assigns a single temperature density to the entire
# loop based on the loop length and the specified heating rate.
rtv = RTVInterface(heating_rate=1e-4*u.Unit('erg cm-3 s-1'))
bundle.load_loop_simulations(rtv)

###########################################################################
# We can then compute the emission as observed by the 171 channel of AIA
# as viewed from an observer at 1 AU directly above the loop.
aia = InstrumentSDOAIA([0, 1]*u.s, pos, pad_fov=(10, 10)*u.arcsec)
maps = aia.observe(bundle, channels=aia.channels[2:3])
maps['171'][0].peek()

###########################################################################
# Additionally, we can look at intensity profiles  along and across the
# loop axis. Note that the intensity is highest at the footpoints because
# we are integrating through the most amount of plasma. Additionally, note
# that the cross-sectional profile has a width consistent with the spatial
# radius we specified when constructing our bundle of strands.
map_coords = all_coordinates_from_map(maps['171'][0])
Tx = map_coords.Tx[21, :]
Ty = map_coords.Ty[:, 58]
with quantity_support():
    plt.figure(figsize=(11, 5))
    plt.subplot(121)
    plt.plot(Tx, maps['171'][0].quantity[21, :])
    plt.subplot(122)
    plt.plot(Ty, maps['171'][0].quantity[:, 58])

###########################################################################
# Finally, we can also compute the AIA 171 intensity as viewed from the
# side in order to see the semi-circular geometry of the loop bundle.
aia = InstrumentSDOAIA([0, 1]*u.s, side_on_view, pad_fov=(10, 10)*u.arcsec)
maps = aia.observe(bundle, channels=aia.channels[2:3])
maps['171'][0].peek()
