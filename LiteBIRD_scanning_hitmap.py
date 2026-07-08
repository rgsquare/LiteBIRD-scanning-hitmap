############################################################
############################################################
# LiteBIRD SCANNING HITMAP
# Raul Gonzalez 7/4/26
############################################################
############################################################
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import astropy.time
import datetime 
from astropy.time import Time
from astropy import coordinates
from astropy.coordinates import (EarthLocation, AltAz, SkyCoord, cartesian_to_spherical ,solar_system_ephemeris)
from astropy.coordinates import (ICRS,get_body_barycentric,BarycentricMeanEcliptic)
from astropy import units as u
##############################
# General constants
pi = np.pi
d2r = pi/180.0
nside = 512
npix = hp.nside2npix(nside)                             # number of pixels
omega_pix = 4.0*np.pi/npix
##############################
# General in-flights parameters
alpha = 37.5;       beta = 57.5;                        # antisun and spin angle inclination (deg)
Ta = 11540.88;      Tb = 1200                           # antisun and spin precession period (seconds) 
omega_a = 2*pi/Ta;  omega_b = 2*pi/Tb                   # antisun and spin angular velocity (rad/s)
d_L2S = 153.59*1e9                                      # Sun-L2 distance (meters)
DL1 = 244450000;    DL2 = 137388000                     # Lissajous semiaxes distances (meters)
TL1 = 15603847.5;   TL2 = 15886593.4                    # Lissajous precession period 1 and 2 (seconds)
psi = -0.8368                                           # Lissajous precession phase (seconds) 
omegaL1 = 2*pi/TL1; omegaL2 = 2*pi/TL2                  # Lissajoux angular velocities (rad/s)
##############################
# Source parameters
# {"source": distance from the sun in meters}
source = {"mercury" : 59.853*1e9 , "venus": 107.59*1e9, "mars" : 250.66*1e9, "jupiter" : 754.05*1e9, "saturn" : 1486.7*1e9, "uranus" : 2953.7*1e9}
source_name = source.keys()
##############################
# Degree to radians transformation
alphar =  alpha * d2r                                   # rad
betar =  beta * d2r                                     # rad
#############################
# Vectors declaration 
maphits = np.zeros(npix)                                                                                                                              
############################
# 3 years observation
solar_system_ephemeris.set("builtin")
t1 = Time('2033-01-01 00:00:01')
t2 = Time('2033-01-01 00:00:02')
dt = t2-t1
time = np.arange(94608000)                               # Time score
a = 10000;                                               # Counter for visualization
############################
# Source selection
print("Select a source for the generation of the hit-map among these: mercury, venus, mars, jupiter, saturn and uranus")
selected_source = input("Which source do you want to simulate? ")
source_name = source.keys()
for i in source_name:
   if i == selected_source:
     d_S = source[selected_source] 
print(f"The selected source is {selected_source}" )
Lissajous = input("Do you want to implement the Lissajous orbit? (yes or not)")
print(f"The generation of the hit-map for {selected_source} has started! ")
####################################################################################
####################################################################################
####################################################################################
# START OF SIMULATION
for t in time:
    ############################################ 
    # Time vector in second
    x = t
    t_vec = t1 + (dt * x)                                                 
    ############################################
    # Ecliptic planet coordinates generation
    icrs_pos = get_body_barycentric(selected_source ,t_vec)
    ra = ICRS(icrs_pos).ra
    dec = ICRS(icrs_pos).dec
    distance = ICRS(icrs_pos).distance
    coord_icrs = SkyCoord(ra, dec, distance, frame='icrs')
    ecl_vec = coord_icrs.transform_to(BarycentricMeanEcliptic) 
    #ecl_vec = (ICRS(icrs_pos).transform_to(BarycentricMeanEcliptic))   #.cartesian.get_xyz().value) to obtain cartesian coord.
    lat = theta = ecl_vec.lat.value 
    lon = phi = ecl_vec.lon.value
    dist = ecl_vec.distance.value
    thetar = theta*d2r
    phir = phi*d2r
    ############################################
    # Ecliptic earth coordinates generation
    earth_coord = coordinates.get_body_barycentric('earth',t_vec,ephemeris='builtin')
    ra_e = ICRS(icrs_pos).ra
    dec_e = ICRS(icrs_pos).dec
    distance_e = ICRS(icrs_pos).distance
    coord_icrs_e = SkyCoord(ra_e, dec_e, distance_e, frame='icrs')
    ecl_vec_e = coord_icrs_e.transform_to(BarycentricMeanEcliptic) 
    lat_e = theta_e = ecl_vec_e.lat.value 
    lon_e = phi_e = ecl_vec_e.lon.value
    dist_e = ecl_vec_e.distance.value
    thetar_e = theta_e*d2r
    phir_e = phi_e*d2r
    ############################################
    theta_a = omega_a * t; 
    theta_b = omega_b * t; 
    x = d_S*np.cos(thetar)*np.cos(phir)                              # Jupiter ecliptic x coordinate
    y = d_S*np.cos(thetar)*np.sin(phir)                              # Jupiter ecliptic y coordinate
    z = d_S*np.sin(thetar)                                           # Jupiter ecliptic z coordinate
    x2 = x*np.cos(phir_e) - y*np.sin(phir_e)                         # Jupiter x coordinate rotated in L2 reference system
    y2 = x*np.sin(phir_e) + y*np.cos(phir_e)                         # Jupiter y coordinate rotated in L2 reference system
    z2 = z                                                           # Jupiter z coordinate rotated in L2 reference system
    V = np.array([[-d_L2S+x2],[y2],[z2]]);                           # Jupiter position vector with respect to L2
    d_L2 = np.linalg.norm(V)/(150.0*1e9)                             # Jupiter distance from L2 in UA
    ######################################################################## 
    # Lissajoux orbit
    VL = np.array([[0],[DL1*np.cos(omegaL1*t)],[DL2*np.sin((omegaL2*t) + psi)]])
    if (Lissajous == "yes"):
       Vfin = V - VL
    else:
       Vfin = V
    ######################################################################## 
    # Rotation around X (antisun axis) 
    R1 = np.array([[1,0,0],[0,np.cos(theta_a),np.sin(theta_a)],[0,-np.sin(theta_a),np.cos(theta_a)]]); 
    # Rotation around Y 
    R2 = np.array([[np.cos(alphar),0,-np.sin(alphar)],[0,1,0],[np.sin(alphar),0, np.cos(alphar)]]); 
    ######################################################################## 
    # Rotation around X (spin axis) 
    R3 = np.array([[1,0,0],[0,np.cos(theta_b),np.sin(theta_b)],[0,-np.sin(theta_b),np.cos(theta_b)]]); 
    # Rotation around Y 
    R4 = np.array([[np.cos(betar),0,-np.sin(betar)],[0,1,0],[np.sin(betar),0, np.cos(betar)]]); 
    ########################################################################
    V1 = np.dot(R1,Vfin);                                                   
    V2 = np.dot(R2,V1);
    V3 = np.dot(R3,V2);
    V4 = np.dot(R4,V3);
    Vf = -V4.T;                                                         # boresight raw vector in the satellite reference system 
    pixel = hp.vec2pix(nside, Vf[0,2],Vf[0,1],-Vf[0,0]);
    ########################################################################
    maphits[pixel] = maphits[pixel]+1;                                  # Hits per pixel vector
    ########################################################################
    if(t == a):
        print(t)
        a = a+10000
####################################################################################
####################################################################################
####################################################################################
# Printing map
print("The hit-map is ready!!")
hit_map = hp.fitsfunc.write_map('Hit_map-'+source+'.fits', maphits)
############################################################################
