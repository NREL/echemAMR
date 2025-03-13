set terminal pdf
set output "ee_lset.pdf"
set termoption font "Helvetica,18"
set key spacing 1.2
set lmargin 8
set rmargin 1
set bmargin 3
set xlabel "x distance (m)"
set ylabel "Color field"
set key top left
plot 'lsetfield_11.dat' u 1:2 w lp ps 1 pt 5 lc 7  lw 1 title "10 cells",\
     'lsetfield_61.dat' u 1:2 w lp ps 1 pt 7 lc 7 lw 1 title "60 cells",\
     'lsetfield_361.dat' u 1:2 w lp ps 1 lc 7  lw 1 title "360 cells",\
     'lsetfield_2161.dat' u 1:2 w lp ps 1 lc 7 lw 1 title "2160 cells"
