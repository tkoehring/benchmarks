# Summit Benchmarking

# Enivronment
---
## Modules

```
module load gcc cuda gdrcopy
module unload darshan-runtime
```

## Variables

```
export MPICH_INSTALL=$HOME/install/mpich
export UCX_INSTALL=$HOME/install/ucx
export OSU_INSTALL=$HOME/install/osu
```

## Directories

```
mkdir install install/mpich install/ucx install/osu
ls install/
	mpich  osu  ucx
```

---
# Software 
---
## MPICH

```
git clone https://github.com/pmodels/mpich.git
cd mpich/
git submodule update --init
./autogen.sh
```

### Building MPICH

```
./configure --with-device=ch4:ucx --with-ucx=$UCX_INSTALL --with-pm=none \
	--with-pmix=$MPI_ROOT --with-cuda=$CUDA_DIR --with-hwloc=embedded \
	CFLAGS=-std=gnu11 --prefix=$MPICH_INSTALL
make -j8
make install -j8
```

## UCX

```
wget https://github.com/openucx/ucx/releases/download/v1.11.0/ucx-1.11.0.tar.gz
tar xvf ucx-1.11.0.tar.gz
```

### Building UCX

```
./configure CC=gcc  CXX=g++ --build=powerpc64le-redhat-linux-gnu --host=powerpc64le-redhat-linux-gnu \
  --with-cuda=$CUDA_DIR --with-gdrcopy=$OLCF_GDRCOPY_ROOT \
  --disable-logging --disable-debug --disable-assertions \
  --prefix=$UCX_INSTALL
make -j8
make install -j8
```

## OSU

```
wget https://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-5.9.tar.gz
tar xvf osu-micro-benchmarks-5.9.tar.gz
```

### Building OSU

```
./configure CC=$MPICH_INSTALL/bin/mpicc CXX=$MPICH_INSTALL/bin/mpicxx \
--enable-cuda --with-cuda-include=$CUDA_DIR/include 
--with-cuda-libpath=$CUDA_DIR/lib64 --prefix=$OSU_INSTALL
make -j8
make install -j8
```

# Interactive Jobs
```
bsub -Is -W 0:30 -nnodes 2 -P csc371 /bin/bash
```

- `-Is` - Indicates an interactive job
- `-W` - Wall time for how long to reserve nodes. Format is `<hours>:<minutes>`
- `-nnodes` - Number of nodes to reserve.
- `-P` - Project to charge time to. MPICH project is `csc371`
- `/bin/bash` - Shell to use for interative job

# Running A Sample Program
```
jsrun -n 2 -r 1 -g 1 --smpiargs="-disable_gpu_hooks" \
    -E UCX_NET_DEVICES=mlx5_0:1 \
    ./test/mpi/pt2pt/pingping \
    -type=MPI_INT -sendcnt=512 -recvcnt=1024 -seed=78 -testsize=4  -sendmem=device -recvmem=device
```

- `jsrun`
	- `-n` - Number of resource sets
	- `-r` - Resources per host
		-  `-n` / `-r` is how many nodes you want to use. This cannot exceed the number allocated from bsub. See [Interactive Jobs](#interactive-jobs)
	-  `-g` - GPU's per resource set
	-  `-E` - Environment setting to set just before exec.
		-  `UCX_NET_DEVICES` - This tells UCX to only use the device and port provided here. Otherwise UCX will try to init all devices and port which is not the correct setup for Summit (no multi-rail setup on Summit)
	- `./test/mpi/pt2pt/pingping` - Application to run
		- Arguments after this are for this application

# Results
```
bash-4.4$ jsrun -n 2 -r 1 -g 1 --smpiargs="-disable_gpu_hooks" -E UCX_NET_DEVICES=mlx5_0:1 ./osu_latency    
[1655933883.766409] [g33n12:337554:0]          parser.c:1885 UCX  WARN  unused env variable: UCX_INSTALL (set UCX_WARN_UNUSED_ENV_VARS=n to suppress this warning)
# OSU MPI Latency Test v5.9
# Size          Latency (us)
0                       1.38
1                       1.38
2                       1.37
4                       1.37
8                       1.36
16                      1.37
32                      1.38
64                      1.48
128                     1.52
256                     1.35
512                     1.44
1024                    1.69
2048                    2.90
4096                    3.46
8192                    4.74
16384                   5.96
32768                   9.71
65536                  12.38
131072                 17.78
262144                 28.34
524288                 49.51
1048576                91.86
2097152               176.55
4194304               346.07
[1655933883.765191] [g33n13:2715108:0]          parser.c:1885 UCX  WARN  unused env variable: UCX_INSTALL (set UCX_WARN_UNUSED_ENV_VARS=n to suppress this warning)

```
