#!/bin/bash
CC_VERSION=(4.8 7 9 10)
CC=gcc
OPTIMIZATIONS=(-O0 -O1 -O2 -O3 -Ofast -Os)

make clean

for VERSION in ${CC_VERSION[@]}; do
    for OPTIMIZATION in ${OPTIMIZATIONS[@]}; do

        if [[ "$VERSION" == "4.8" ]]
        then
            # add functionality to add c99 flag
        else
            FLAGS="$OPTIMIZATION"
        fi

        make CC=$CC-$VERSION CFLAGS="$FLAGS" all
        make CC=$CC-$VERSION CFLAGS="$FLAGS" install
        ./bin/yaksa/pack_unpack -r 100 -M 32 -i H -o H > results/pack_unpack-$VERSION$OPTIMIZATION.data
        make clean

        DEBUG="$FLAGS -DNDEBUG"
        make CC=$CC-$VERSION CFLAGS="$DEBUG" all
        make CC=$CC-$VERSION CFLAGS="$DEBUG" install
        ./bin/yaksa/pack_unpack -r 100 -M 32 -i H -o H > results/pack_unpack-$VERSION$OPTIMIZATION-NDEBUG.data
        make clean
    done
done


