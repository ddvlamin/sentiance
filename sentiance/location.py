"""
This module provides a Location class that represents a lat long place

The locations are encoded with a precision specified by the number of bits

Locations that are close enough are considered the same

There is also a class UserLocations that creates a set of user locations based on a csv file
Locations that are similar according to the precision are collapsed and regarded as one and the same location

Example:

  .. code-block:: bash

    $python location.py "../data/Copy of person.1.csv" 51.209335 4.3883753
    (51.209335, 4.3883753) found in set

    $python location.py "../data/Copy of person.1.csv" 51.209325 4.3883763
    (51.209325, 4.3883763) found in set

    $python location.py "../data/Copy of person.1.csv" 50.209325 4.3883763
    (50.209325, 4.3883763) not found in set (maybe decrease precision?)

"""

import sys
import argparse

from math import radians, cos, sin, asin, sqrt

import numpy as np
import pandas as pd

POLAR_CIRCUMFERENCE = 40007863
EQUATOR_CIRCUMFERENCE = 40075017
NBITS = 34

def decode_geohash(geohash, nbits):
  """
  Computes the latitude and longitude of the given geohash assuming a nbits precision

  :param geohash: geohash computed by the encode method of a Location object
  :param nbits: precision of the geohash in bits

  :returns Location: location with the lat and long of the decoded geohash
  """
  binhash = bin(geohash)[2:]
  prefix = '0' * (nbits-len(binhash))
  binhash = prefix + binhash

  minLat = -90
  maxLat = 90
  minLng = -180
  maxLng = 180    

  for i, bit in enumerate(binhash):
    if i%2 == 0:
      midpoint = (minLng + maxLng) / 2
      if bit == '0':
        maxLng = midpoint
      else:
        minLng = midpoint
    else:
      midpoint = (minLat + maxLat) / 2
      if bit == '0':
        maxLat = midpoint
      else:
        minLat = midpoint

  resultLng = (minLng + maxLng) / 2
  resultLat = (minLat + maxLat) / 2

  return Location(resultLat, resultLng, nbits)

def haversine(loc1, loc2):
  """
  Calculate the great circle distance between two points 
  on the earth (specified in decimal degrees)
  
  from https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points

  :param loc1: first location
  :type loc1: Location
  :param loc2: second location
  :type loc2: Location

  :returns: earth distance between loc1 and loc2 in kilometers

  >>> gent = Location(51.054340, 3.717424)
  >>> apen = Location(51.219448, 4.402464)
  >>> 50 < haversine(gent, apen) < 70
  True
  """
  lat1, lon1 = loc1
  lat2, lon2 = loc2

  # convert decimal degrees to radians 
  lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

  # haversine formula 
  dlon = lon2 - lon1 
  dlat = lat2 - lat1 
  a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
  c = 2 * asin(sqrt(a)) 
  r = 6371 # Radius of earth in kilometers. Use 3956 for miles
  return c * r
  
def lat_error(nbits):
  """
  Computes the maximum latitude error according to the number of bits used for encoding

  Example:

    16 bits for latitude has an 305 meter accuracy
    20 bits for latitude has an 19 meter accuracy

  :param nbits: number of bits used to encode latitude

  :returns tuple: (maximumr error in meters, maximum error in degrees)

  >>> 18 < lat_error(20)[0] < 20
  True
  """
  one_degree_dist = POLAR_CIRCUMFERENCE/180
  max_error = 90*(2**-nbits)
  return (one_degree_dist*max_error), max_error

def long_error(lat, nbits):
  """
  Computes the maximum longitude error according to the number of bits used for encoding

  Note: the accuracy depends on the latitude (as the earth's circumference is smaller
  at the non-zero latitudes than it is at the equator)
  
  :param loc: location for which to compute the maximum longitude encoding error
  :type Location:

  :returns tuple: (maximum error in meters, maximum error in degrees)

  >>> err1 = long_error(51, 15)[0]
  >>> err2 = long_error(60, 15)[0]
  >>> err1 > err2
  True
  >>> err2 = long_error(51, 10)[0]
  >>> err1 < err2
  True
  """
  one_degree_dist = (EQUATOR_CIRCUMFERENCE/360)*np.cos(np.deg2rad(lat))
  max_error = 180*(2**-nbits)
  return (one_degree_dist*max_error), max_error

class Location(tuple):
  """
  This class represents a Location tuple with latitude and longitude coordinates
  It uses an encoding with nbits precision to test whether two difference locations are the same or note

  """
  
  def __new__(cls, latitude, longitude, nbits=40):
    return super().__new__(cls, (latitude, longitude))
  
  def __init__(self, latitude, longitude, nbits=40):
    """
    :param latitude: latitude in decimal degrees
    :type latitude: float
    :param longitude: longitude in decimal degrees
    :type longitude: float
    :param nbits: precision in bits for the lat long encoding (see :func:`sentiance.location.Location.lat_error` for more details on the meaning of nbits precision)
    :type nbits: int
    """
  
    if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
      raise ValueError("latitude should be between -90 and 90 degrees, longitude between -180 and 180 degrees")

    self.__lat = latitude
    self.__lng = longitude
    self.__nbits = nbits

  @property
  def lat(self):
    return self.__lat

  @property
  def lng(self):
    return self.__lng
  
  @property
  def nbits(self):
    return self.__nbits
  
  @property
  def lat_error(self):
    """
    see :func:`sentiance.location.lat_error`
    """
    lat_nbits = np.floor(self.nbits/2)
    return lat_error(lat_nbits)

  @property
  def long_error(self):
    """
    see :func:`sentiance.location.long_error`
    """
    lng_nbits = np.ceil(self.nbits/2)
    return long_error(self.lat, lng_nbits)

  @property
  def geohash(self):
    if not hasattr(self, "__geohash"):
      self.__geohash = self.__encode()
    return self.__geohash
 
  def distance(self, loc):
    """
    Computes distance from curent point to *loc*

    :param loc: location to compute distance to
    :return: distance in kilometers
    """
    return haversine(self, loc)

  def __encode(self):  
    """
    Computes a hash based on the latitude and longitude

    The hash for nearby points should also be close together

    The precision depends on the number of bits specified

    The computation is based on:
      https://www.factual.com/blog/how-geohashes-work/
      https://en.wikipedia.org/wiki/Geohash

    TODO: there are faster ways to compute the hash only using bit operations

    :returns: a hash for latitude and longitude
 
    >>> hoboken = Location(51.17165565490723, 4.346981048583984, 36)
    >>> neighbor1 = Location(51.1720, 4.3476, 36)
    >>> outside = Location(51.1716,4.3459, 36)
    >>> hoboken.geohash == neighbor1.geohash
    True
    >>> hoboken.geohash == outside.geohash
    False

    """
    minLat = -90
    maxLat = 90
    minLng = -180
    maxLng = 180

    result = 0

    for i in range(self.nbits):
      if (i % 2 == 0):                
        midpoint = (minLng + maxLng) / 2
        if (self.lng < midpoint):
          result <<= 1            
          maxLng = midpoint
        else:
          result = result << 1 | 1
          minLng = midpoint
      else:
        midpoint = (minLat + maxLat) / 2
        if (self.lat < midpoint):
          result <<= 1             
          maxLat = midpoint
        else:
          result = result << 1 | 1
          minLat = midpoint
 
    return result
  
  def __hash__(self):
    """
    Computes encoding so that lat and long coordinates that are close to each other (in terms of precision) might have the same encoding
    :return: lat and long encoding 
    """
    return self.geohash
  
  def __eq__(self, loc):
    """
    Test whether this location is the same as loc based on the geohash, so lat and long coordinates that are close
    to each other (in terms of precision) might have the same encoding and test equal

    >>> hoboken_high = Location(51.17165565490723, 4.346981048583984, 36)
    >>> hoboken_low = Location(51.17165565490723, 4.346981048583984, 30)
    >>> hoboken_low == hoboken_high
    Traceback (most recent call last):
      ...
    ValueError: can't compare locations of different precision 30 and 36
    """
    if self.nbits != loc.nbits:
      raise ValueError(f"can't compare locations of different precision {self.nbits} and {loc.nbits}")
    return loc.geohash == self.geohash
  
  def bounding_box(self):
    """
    Computes a bounding box around the encoding of the lat long, the box has width equal
    to the max long error and the height is equal to the max lat error

    :return: tuple of four elements:
      (lower side of rectangle,
      upper side of rectangle,
      left side of rectangle,
      right side of rectangle)
    """ 
    loc = decode_geohash(self.geohash, self.nbits)
    _, max_lat_error = loc.lat_error
    _, max_lng_error = loc.long_error
    
    return (
      loc.lat - max_lat_error,
      loc.lat + max_lat_error,
      loc.lng - max_lng_error,
      loc.lng + max_lng_error
    )
  
  def bounding_corners(self):
    """
    Computes the bounding box but returns the box specified as corners of Location type
    :return: tuple of four Location objects starting from lowerleft corner, clockwise
    """
    lb, ub, left, right = self.bounding_box()
    
    return (
        Location(lb, left),
        Location(ub, left),
        Location(ub, right),
        Location(lb, right)
    )

class UserLocations(set):
  """
  A special set of a user's Location objects where locations are deemed the same when they have the same geohash (although they
  might have different lat long coordinates)

  TODO: implement other set operators
  """
  def __init__(self, filepath, nbits, sep=";"):
    """
    Constructs a set of locations stored in the user's location file

    :param filepath: path to the file that contains the user's location, should be csv file with columns latitude and longitude
    :param nbits: bit precision for encoding the lat long locations (influences which locations are considered the same)
    """
    self._nbits = nbits
    df = pd.read_csv(filepath, sep=sep)

    for row in df.to_dict(orient="records"):
      loc = Location(row["latitude"], row["longitude"], self._nbits)
      self.add(loc)

  def __contains__(self, tpl):
    """
    Checks whether a location with 'similar' lat long coordinates exists in the set

    :param tpl: (latitude,longitude) tuple in decimal degrees
    """
    loc = Location(tpl[0], tpl[1], self._nbits)
    return super().__contains__(loc)

  def add(self, tpl):
    """
    Adds a location object with the lat long coordinates in the tuple tpl to the set

    :param tpl: (latitude, longitude) tuple in decimal degrees
    """
    loc = Location(tpl[0], tpl[1], self._nbits)
    super().add(loc)

if __name__ == "__main__":
  import doctest
  doctest.testmod()

  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("datafile",help="csv file with latitude and longitude columns of the user's locations")
  parser.add_argument("latitude", type=float, help="latitude coordinate you want to check in the user's locations")
  parser.add_argument("longitude", type=float, help="longitude coordinate you want to check in the user's locations")
  parser.add_argument("--precision", type=int, default=NBITS, help="precision of location encoding, default is 34 or approximately a max latitude error of roughly 150 meters")

  args = parser.parse_args(sys.argv[1:])
  datafile = args.datafile
  latitude = args.latitude
  longitude = args.longitude
  precision = args.precision

  locations = UserLocations(datafile, precision)
  user_location = Location(latitude, longitude, precision)

  if user_location in locations:
    print(f"{user_location} found in set")
  else:
    print(f"{user_location} not found in set (maybe decrease precision?)")

