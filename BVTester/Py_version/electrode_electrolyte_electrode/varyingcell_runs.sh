#!/bin/bash
for n in 11 61 361 2161
do
    python trode_lyte_trode.py $n 0.001 0.24 0.74 0 >> outnew
done
