set terminal png
set output "cea_soln_2d.png"
set termoption font "Helvetica,16"
set title "Tilted 2D CEA case"
set key spacing 1.2
set lmargin 9
set rmargin 2
set bmargin 3
set xlabel "x distance (m)"
set ylabel "Potential (V)"
set key at 1.07,0.1
#set yrange [-0.7:0.2]
set ytics 0.2
set xrange [0:1.5]
plot 'exactsoln_181.dat' u 1:2 w lp ps 3 pt 8 lc 8 pi 6 lw 2 title "1d solution",\
     'pot_181.dat' u 1:2 w lp ps 2 pt 6 lc 6 lw 3 pi 10 title "128 x 128"
