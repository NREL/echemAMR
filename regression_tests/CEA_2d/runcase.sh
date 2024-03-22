#!/bin/bash
rm *.dat
$1 -n $2 ./*.ex inputs_highcond amr.plot_file=plt
finfile=$(ls -d plt?????/ | tail -n 1)
mv ${finfile} finalplt
. ../clean.sh
