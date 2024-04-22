#!/bin/bash
rm *.dat
$1 -n $2 ./*.ex inputs_x amr.plot_file=plt
