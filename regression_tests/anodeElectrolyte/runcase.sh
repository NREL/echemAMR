#!/bin/bash
$1 -n $2 ./*.ex inputs_x amr.plot_file=plt
finfile=$(ls -d plt?????/ | tail -n 1)
mv ${finfile} finalplt_x
. ../clean.sh
$1 -n $2 ./*.ex inputs_y amr.plot_file=plt
finfile=$(ls -d plt?????/ | tail -n 1)
mv ${finfile} finalplt_y
. ../clean.sh
$1 -n $2 ./*.ex inputs_z amr.plot_file=plt
finfile=$(ls -d plt?????/ | tail -n 1)
mv ${finfile} finalplt_z
. ../clean.sh
