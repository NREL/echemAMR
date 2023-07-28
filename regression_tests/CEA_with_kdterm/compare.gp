set yrange [-1:0.3]
plot 'echemsoln_x.dat' u 1:2 w lp pt 4 pi 5 ps 2 title "echem x solution",\
     'echemsoln_y.dat' u 1:2 w lp pt 5 pi 7 ps 2 title "echem y solution",\
     'echemsoln_z.dat' u 1:2 w lp pt 6 pi 11 ps 2 title "echem z solution",\
     'soln_fd_257.dat' u 1:2 w lp pt 7 pi 13 ps 2 title "python fd solution"

