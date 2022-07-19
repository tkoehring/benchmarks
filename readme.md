# Yaksa Micro Benchmarks

## TODO
Below is a list of existing benchmark programs and any TODO items left for them.

1. [memcpy.c](src/gnu/memcpy.c) -- Standard gnu memcpy
    - Complete
2. [cuda_memcpy.c](src/cuda/cuda_memcpy.c) -- Standard cuda memcpy
    - Complete
2. [pack_unpack.c](src/yaksa/pack_unpack.c) -- Yaksa Pack and Unpack Contiguous
   Buffer
    - Complete
4. [pack_unpack_nc.c](src/yaksa/pack_unpack_nc.c) -- Yaksa Pack Unpack
   Non-Contiguous Buffer
    - Determine reason that `Host -> Device` or `Device -> Host` configuration
      does not work but `Host -> Host` or `Device -> Device` do work.

All of these programs are considered to be working, though some fine tuning for
adding parameter arguments may be neccesary.

## Building
All source code can be found in `src/` and are organized based on the library
they use. 

There is a [Makefile](Makefile) including in this repository that will compile
and build all of the programs. Currently, you will need to add the yaksa
installed library location to `LD_LIBRARY_PATH` for the compilation to work, as
well as editing the `Makefile` to reflect the install locations of `yaksa` and
`cuda` (this is commented in the Makefile itself). Additionally, you will need
to have `nvcc`'s location added to your `$PATH` environment variable.

To compile and build, run the following:

```
make all
make install
```

The executables will be stored in the newly created `bin` directory, in the same
fashion they are in the `src` folder.

`make clean` is included and works as expected.

## Running
Below is instructions on how to run each benchmark program.

### `memcpy.c`
This program copies data from one buffer to another.

Arguments:
- `-M` - The size in megabytes the buffer should be.
- `-r` - The number of runs that should be performed. Timings are averaged over
         this number.

To run:

```
./bin/gnu/memcpy
```

### `cuda_memcpy.c`
This program copies data from one GPU buffer to another.

Arguments:
- `-M` - The size in megabytes the buffer should be.
- `-r` - The number of runs that should be performed. Timings are averaged over
         this number.

To run:

```
./bin/cuda/cuda_memcpy
```

### `pack_unpack.c`
This program packs and unpacks data from a user defined buffer.
Arguments:
- `-M` - The size in megabytes the buffer should be.
- `-r` - The number of runs that should be performed. Timings are averaged over
         this number.
- `-i <H || D>` - Indicates whether the input buffer is on the host (cpu) or
  device (gpu), ie. `-i H` indicates a cpu input buffer.
- `-o <H || D>` - Indicates whether the output buffer is on the host (cpu) or
  device (gpu).

To run:

```
./bin/yaksa/pack_unpack -r <runs> -M <size> -i <input type> -o <output type>
```

### `pack_unpack_nc.c`
This program packs and unpacks data from a user defined buffer, non-contiguous
buffer.
Arguments:
- `-b` - The block size for the vector type.
- `-M` - The size in megabytes the buffer should be.
- `-r` - The number of runs that should be performed. Timings are averaged over
         this number.
- `-s` - The stride length for the vector type.
- `-i <H || D>` - Indicates whether the input buffer is on the host (cpu) or
  device (gpu), ie. `-i H` indicates a cpu input buffer.
- `-o <H || D>` - Indicates whether the output buffer is on the host (cpu) or
  device (gpu).
To run:

```
./bin/yaksa/pack_unpack_nc -r <runs> -M <size> -b <block length> -s <stride
length> -i <input type> -o <output type>
```
