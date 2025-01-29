#!/bin/bash
for n in 11 61 361 2161
do
    python trode_lyte.py $n 0.001 0.54 >> outnew
done
