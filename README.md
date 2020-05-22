
Question 1 - Data lookup:

I've created a class Location which is a immutable tuple containing latitude and longitude coordinates

The latitue and longitude are encoded into a single number so that locations that are close to each other also encode to a number that is close.

You can also specify an encoding precision. Depending on that precision it will then consider locations that are close enough as the same

There is also a class UserLocations which is basically a set of Locations objects that is created from the specified file, again speficying a precision to define which locations are considered the same 

The computation is based on:
  https://www.factual.com/blog/how-geohashes-work/
  https://en.wikipedia.org/wiki/Geohash

You can invoke the script like (--help for information about the input arguments)

  .. code-block:: bash

    $python location.py "../data/Copy of person.1.csv" 51.209335 4.3883753
    (51.209335, 4.3883753) found in set

Complexity

  The computation of the geohash is linear (O(nbits)) in the number of bits, the implementation itself could be done faster by using bit operations only
  To build up a set of user locations each record in the csv file of user locations needs to be processed and the geohash computed, so this is linear in the number of locations, i.e. O(nbits * locations)
  Once the set of user locations is built, you can do a lookup in constant time, depending on the number of precision bits that need to be computed, i.e. O(nbits) or O(1) if we consider this constant
  Memory complexity: depends on the precision, the lower the precision the more locations are collapsed on one another. Per record that is stored one only needs to reserve memory for storing the Location object (latitude, longitude, nbits, geohash). So the complexity is sublinear if the precision in not to high. There is of course also overhead for keeping the set structure

Question 2 - Home-Work:

  The algorithm:
  The goal is to compute some features per location, where location is determined by the precision of the Location object (as discussed in Question 1). So samples with lat long coordinates that are close enough will be mapped to the same location.

  Per sample we first add some columns/features: 
    - date
    - day of the week that the person visited the location, 
    - hour of the day that the person arrived at the location
    - duration in hours that the person stayed here
    - geohash: the location encoding with precision of 30 bits

  This step is linear in the number of samples

  In the next step we group all samples together that map to the same location/geohash. Per group we calculate a new sample with the following features:
    - day frequency: how frequent does the user visit the place
    - mean duration: how long does he/she spend at that location on average
    - mean starthour: on average at what hour of the day does he/she arrive at the location
    - mean latitude
    - mean longitude
    - number_of_days: compute how many of the 7 weekdays that he/she spends more than MIN_DAY_DENSITY (=0.5) of the time at that location (e.g. for home this will likely be close to 7 days)

  This step should also be linear in the number of samples (and reduces the data set as certain samples of similar location are collapsed)

  We filter out locations where the user spends less than 1 hour and that have a day frequency that is less than the average
 
  This leaves very few locations. Then we check that all these remaining locations (in the order of 5) that all the distances are within reason (200km), the locations that fall outside that range are removed. The computations of these distances have quadratic complexity, but as the number of remaining samples are very low, this can be ignored.

  The locations that have a number of day visits more than or equal to 6 are classified as home, the others as work

  I've also applied a clustering algorithm MeanShift that seems to find clusters that contain almost the same home work locations. the complexity of meanshift according to documentation is O(T*n*log(n)) where T is the dimension and n the samples

