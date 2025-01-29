set terminal pdf
set output "eee_newton_errconv.pdf"
set termoption font "Helvetica,18"
set key spacing 1.2
#set lmargin 12.5
#set rmargin 3
#set bmargin 3
set xlabel "Number of Iterations"
set ylabel "Residual error"
set xrange [0:6]
set yrange [1e-16:1000]
set ytics (1,1e-4,1e-8,1e-12,1e-16)
set key top right
#set xtics format "%5.2e"
set ytics format "%5.2e"
set logscale y 
plot 'errconv_11.dat' u 1:2 w lp ps 1 pt 6 lc 8  lw 2 title "10 cells",\
     'errconv_61.dat' u 1:2 w lp ps 1 lw 2 lc 7 title "60 cells",\
     'errconv_361.dat' u 1:2 w lp ps 1 lw 2 lc 2 title "360 cells",\
     'errconv_2161.dat' u 1:2 w lp ps 1 lw 2 lc 6 title "2160 cells"
