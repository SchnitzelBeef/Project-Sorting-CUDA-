## Validating the Implementation

To validate the implementation, navigate to `/code/cuda/radix/` and configure the following parameters in the `Makefile`:

 - `RADIX_Q` - Number of elements processed by each thread
 - `RADIX_B` - Block size
 - `BITS` - Number of bits processed per pass
 - `N` - Total number of elements

You can validate the program by running `make`, which will compile and execute the validation for the specified parameters. By default, the program automatically generates an input array of size `N`, where each element is a 32-bit integer formed by concatenating four randomly generated 8-bit values (bytes).

### Using a Custom Input

To use your own input, ensure it is formatted as a Futhark array of unsigned 32-bit integers, with each element suffixed by `u32`. Save this array to a file named `input.txt` inside the `/code/` directory, replacing the existing array definition. Then, return back into `/code/cuda/radix/` and execute:

```
./radix N 1
```

where `N` represent the number of elements in the given array.
