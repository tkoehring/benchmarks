#!/bin/bash
CC_VERSION=(4.8 7 9 10)
CC=gcc
OPTIMIZATIONS=(-O0 -O1 -O2 -O3 -Ofast -Os)

make clean

for VERSION in ${CC_VERSION[@]}; do
    for OPTIMIZATION in ${OPTIMIZATIONS[@]}; do

        if [[ "$VERSION" == "4.8" ]]
        then
            FLAGS="$OPTIMIZATION -std=c99"
        else
            FLAGS="$OPTIMIZATION"
        fi

        make CC=$CC-$VERSION YAKSA_CFLAGS="$FLAGS" yaksa
        make CC=$CC-$VERSION YAKSA_CFLAGS="$FLAGS" install_yaksa
        ./bin/yaksa/pack_unpack > results/pack_unpack-$VERSION$OPTIMIZATION.data
        make clean

        DEBUG="$FLAGS -DNDEBUG"
        make CC=$CC-$VERSION YAKSA_CFLAGS="$DEBUG" yaksa
        make CC=$CC-$VERSION YAKSA_CFLAGS="$DEBUG" install_yaksa
        ./bin/yaksa/pack_unpack > results/pack_unpack-$VERSION$OPTIMIZATION-NDEBUG.data
        make clean
    done
done


