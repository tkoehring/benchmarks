#!/bin/bash
CC_VERSION=(7 9 10)
CC=gcc

OPTIMIZATIONS=(-O0 -O1 -O2 -O3)

make clean

for VERSION in ${CC_VERSION[@]}; do
    for OPTIMIZATION in ${OPTIMIZATIONS[@]}; do
        make CC=$CC-$VERSION YAKSA_CFLAGS=$OPTIMIZATION yaksa
        make CC=$CC-$VERSION YAKSA_CFLAGS=$OPTIMIZATION install_yaksa
        ./bin/yaksa/pack_unpack > results/pack_unpack-$VERSION$OPTIMIZATION.data
        make clean
    done
done


