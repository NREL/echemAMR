set terminal pdf
set output "eee_conv.pdf"
set termoption font "Helvetica,18"
set key spacing 1.2
#set lmargin 12.5
#set rmargin 3
#set bmargin 3
set xlabel "Number of cells"
set ylabel "Error L2 norm"
set key top right
#set xtics format "%5.2e"
set ytics format "%5.2e"
set log
set xrange [8:3000]
plot 'outnew' u 1:2 w lp  lw 2 ps 1 pt 5 lc 6 title "Immersed Interface Scheme",\
    'outnew' u 1:(1/$1) w l  lw 3 lc 8 title "First order"
