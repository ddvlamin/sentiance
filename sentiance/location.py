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

import numpy as np
import pandas as pd

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
        lat_nbits = np.floor(self._nbits/2)
        one_degree_dist = Location.polar_circumference/180
        max_error = 90*(2**-lat_nbits)
        return (one_degree_dist*max_error), max_error

    def long_accuracy(self):
        lng_nbits = np.ceil(self._nbits/2)
        one_degree_dist = (Location.equator_circumference/360)*np.cos(np.deg2rad(self._lat))
        max_error = 180*(2**-lng_nbits)
        return (one_degree_dist*max_error), max_error
    
    def encode(self):    
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

    def decode(self):
        if not hasattr(self, "_geohash"):
            self.encode()
            
        if not hasattr(self, "_hashpoint"):
            binhash = bin(self._geohash)[2:]
            prefix = '0' * (self._nbits-len(binhash))
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

            self._hashpoint = Location(resultLat, resultLng, self._nbits)
        
        return self._hashpoint
    
    def bounding_box(self):
        self.decode()
        _, max_lat_error = self.lat_accuracy()
        _, max_lng_error = self.long_accuracy()
        
        return (
            self._hashpoint._lat - max_lat_error,
            self._hashpoint._lat + max_lat_error,
            self._hashpoint._lng - max_lng_error,
            self._hashpoint._lng + max_lng_error
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
