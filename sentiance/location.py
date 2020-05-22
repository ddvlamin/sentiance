"""
NBITS = 36
hoboken = Location(51.17165565490723, 4.346981048583984, NBITS)
neighbor1 = Location(51.1720, 4.3476, NBITS)
outside = Location(51.1716,4.3459, NBITS)

(47.92519731131873, 0.0006866455078125)
(76.3089427947998, 0.00034332275390625)
(51.17156982421875, 51.17225646972656, 4.346466064453125, 4.34783935546875)
((51.17156982421875, 4.346466064453125), (51.17225646972656, 4.346466064453125), (51.17225646972656, 4.34783935546875), (51.17156982421875, 4.34783935546875))

locations = set()
locations.add(hoboken)
assert neighbor1 in locations
assert outside not in locations

"""

from math import radians, cos, sin, asin, sqrt

import numpy as np
import pandas as pd

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

class Location(tuple):
  """
  This class represents a Location tuple with latitude and longitude coordinates
  It uses an encoding with nbits precision to test whether two difference locations are the same or note
  """
  POLAR_CIRCUMFERENCE = 40007863
  EQUATOR_CIRCUMFERENCE = 40075017
  
  def __new__(cls, latitude, longitude, nbits=40):
    return super().__new__(cls, (latitude, longitude))
  
  def __init__(self, latitude, longitude, nbits=40):
    """
    :param latitude: latitude in decimal degrees
    :type latitude: float
    :param longitude: longitude in decimal degrees
    :type longitude: float
    :param nbits: precision in bits for the lat long encoding (see :func:`sentiance.location.Location.lat_accuracy` for more details on the meaning of nbits precision)
    :type nbits: int
    """
    self._lat = latitude
    self._lng = longitude
    self._nbits = nbits
  
  def lat_accuracy(self):
    """
    Computes the latitude accuracy according to the number of bits used for encoding

    32 bits (or 16 bits for latitude) has an 305 meter accuracy
    40 bits (or 20 bits for latitude) has an 19 meter accuracy

    :returns tuple: (maximumr error in meters, maximum error in degrees)
    """
    lat_nbits = np.floor(self._nbits/2)
    one_degree_dist = Location.POLAR_CIRCUMFERENCE/180
    max_error = 90*(2**-lat_nbits)
    return (one_degree_dist*max_error), max_error

  def long_accuracy(self):
    """
    Computes the longitude accuracy according to the number of bits used for encoding

    Note: the accuracy depends on the latitude (as the earth's circumference is smaller
    at the non-zero latitudes than it is at the equator)

    :returns tuple: (maximumr error in meters, maximum error in degrees)
    """
    lng_nbits = np.ceil(self._nbits/2)
    one_degree_dist = (Location.EQUATOR_CIRCUMFERENCE/360)*np.cos(np.deg2rad(self._lat))
    max_error = 180*(2**-lng_nbits)
    return (one_degree_dist*max_error), max_error
 
  def distance(self, loc):
    """
    Computes distance from curent point to *loc*

    :param loc: location to compute distance to
    :return: distance in kilometers
    """
    return haversine(self, loc)

  def encode(self):  
    """
    Computes a hash based on the latitude and longitude

    The hash for nearby points should also be close together

    The precision depends on the number of bits specified

    The computation is based on:
      https://www.factual.com/blog/how-geohashes-work/
      https://en.wikipedia.org/wiki/Geohash

    TODO: there are faster ways to compute the hash only using bit operations

    :returns: a hash for latitude and longitude
    """
    if not hasattr(self, "_geohash"):
      minLat = -90
      maxLat = 90
      minLng = -180
      maxLng = 180

      result = 0

      for i in range(self._nbits):
        if (i % 2 == 0):                
          midpoint = (minLng + maxLng) / 2
          if (self._lng < midpoint):
            result <<= 1            
            maxLng = midpoint
          else:
            result = result << 1 | 1
            minLng = midpoint
        else:
          midpoint = (minLat + maxLat) / 2
          if (self._lat < midpoint):
            result <<= 1             
            maxLat = midpoint
          else:
            result = result << 1 | 1
            minLat = midpoint

      self._geohash = result
                  
    return self._geohash
  
  def __hash__(self):
    """
    Computes encoding so that lat and long coordinates that are close to each other (in terms of precision) might have the same encoding
    :return: lat and long encoding 
    """
    enc = self.encode()
    return enc
  
  def __eq__(self, loc):
    """
    Test whether this location is the same as loc based on the geohash, so lat and long coordinates that are close
    to each other (in terms of precision) might have the same encoding and test equal
    """
    loc_enc = loc.encode()
    return loc_enc == self.encode()
  
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
    if not hasattr(self, "_geohash"):
      self.encode()
        
    loc = decode_geohash(self._geohash, self._nbits)
    _, max_lat_error = loc.lat_accuracy()
    _, max_lng_error = loc.long_accuracy()
    
    return (
      loc._lat - max_lat_error,
      loc._lat + max_lat_error,
      loc._lng - max_lng_error,
      loc._lng + max_lng_error
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
    def __init__(self, filepath, nbits, sep=";"):
        self._nbits = nbits
        df = pd.read_csv(filepath, sep=sep)

        for row in df.to_dict(orient="records"):
            loc = Location(row["latitude"], row["longitude"], self._nbits)
            self.add(loc)

    def __contains__(self, tpl):
      loc = Location(tpl[0], tpl[1], self._nbits)
      return super().__contains__(loc)

    def add(self, tpl):
      loc = Location(tpl[0], tpl[1], self._nbits)
      super().add(loc)

class UserLocationMap(dict):
    def __init__(self, filepath, nbits, sep=";"):
        self._nbits = nbits
        df = pd.read_csv(filepath, sep=sep)

        for row in df.to_dict(orient="records"):
            loc = Location(row["latitude"], row["longitude"], self._nbits)
            try:
              self[loc].append(row)
            except KeyError:
              self[loc] = [row]

    def __contains__(self, tpl):
      loc = Location(tpl[0], tpl[1], self._nbits)
      return super().__contains__(loc)
    
    def __getitem__(self, tpl):
      loc = Location(tpl[0], tpl[1], self._nbits)
      return self[loc]

    def __setitem__(self, tpl, value):
      raise NotImplementedError("set item not implemented")
