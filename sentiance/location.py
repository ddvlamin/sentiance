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
    polar_circumference = 40007863
    equator_circumference = 40075017
    
    def __new__(cls, latitude, longitude, nbits=40):
        return super().__new__(cls, (latitude, longitude))
    
    def __init__(self, latitude, longitude, nbits=40):
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
        one_degree_dist = Location.polar_circumference/180
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
        one_degree_dist = (Location.equator_circumference/360)*np.cos(np.deg2rad(self._lat))
        max_error = 180*(2**-lng_nbits)
        return (one_degree_dist*max_error), max_error
   
    def distance(self, loc):
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
        enc = self.encode()
        return enc
    
    def __eq__(self, loc):
        loc_enc = loc.encode()
        return loc_enc == self.encode()

    
    def bounding_box(self):
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
        Starting lowerleft corner, clockwise
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
