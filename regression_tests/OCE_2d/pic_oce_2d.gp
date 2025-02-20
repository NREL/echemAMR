set terminal png
set output "oce_soln_2d.png"
set termoption font "Helvetica,16"
set title "2D open-circuit electrode"
set key spacing 1.2
set lmargin 9
set rmargin 1
set bmargin 3
set xlabel "Distance (m)"
set ylabel "Potential (V)"
set key at 0.7,0.15
set yrange [-0.05:0.25]
plot 'exactsoln_128.dat' u 1:2 w lp ps 2 pt 8 lc 8 pi 6 lw 2 title "Exact",\
     'pot_128.dat' u 1:2 w lp ps 1 pt 6 lc 6 lw 2 pi 5 title "128 x 128"
