
Question 1 - Data lookup:

I've created a class Location which is a immutable tuple containing latitude and longitude coordinates

You can also specify an encoding precision. Depending on that precision it will then consider locations that are close enough as the same

There is also a class UserLocations which is basically a set of Locations objects that is created from the specified file, again speficying a precision to define which locations are considered the same 

You can invoke the script like (--help for information about the input arguments)

  .. code-block:: bash

    $python location.py "../data/Copy of person.1.csv" 51.209335 4.3883753
    (51.209335, 4.3883753) found in set

Complexity

  The computation of the geohash is linear (O(nbits)) in the number of bits, the implementation itself could be done faster by using bit operations only
  To build up a set of user locations each record in the csv file of user locations needs to be processed and the geohash computed, so this is linear in the number of locations, i.e. O(nbits * locations)
  Once the set of user locations is built, you can do a lookup in constant time, depending on the number of precision bits that need to be computed, i.e. O(nbits) or O(1) if we consider this constant
