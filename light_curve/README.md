# Light Curve Operations

## Code Author

Chris Shallue: [@cshallue](https://github.com/cshallue)

## Python modules

* `binning`: Utility for binning and aggregating points in a light curve.
* `kepler_io`: Functions for reading Kepler data.
* `periodic_event`: Event class, which represents a periodic event in a light curve.
* `util`: Light curve utility functions.

## Fast ops

The [fast_ops](fast_ops/) subdirectory contains optimized C++ light curve
operations. These operations can be compiled for Python using
[CLIF](https://github.com/google/clif). The [fast_ops/python](fast_ops/python/)
directory contains CLIF API description files.
