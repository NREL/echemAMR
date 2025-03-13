set terminal pdf
set output "ee_solutions.pdf"
set termoption font "Helvetica,18"
set key spacing 1.2
set lmargin 8
set rmargin 1
set bmargin 3
set xlabel "x distance (m)"
set ylabel "Potential (V)"
#set key top center
#set yrange [-1.0:1.0]
plot 'soln_exact.dat' u 1:2 w lp ps 1 pi 50 pt 6 lw 2 title "Exact",\
     'soln_fd_11.dat' u 1:2 w lp ps 1  lw 2 title "10 cells",\
     'soln_fd_61.dat' u 1:2 w lp ps 1  pi 10 lw 2 title "60 cells",\
     'soln_fd_361.dat' u 1:2 w lp ps 1 pi 40 lw 2 title "360 cells",\
     'soln_fd_2161.dat' u 1:2 w l lw 2 lc 8 title "2160 cells",\
