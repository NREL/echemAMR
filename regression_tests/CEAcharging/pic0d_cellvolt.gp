set terminal png
set output "zerodsoln_cellvolt_1d.png"
set termoption font "Helvetica,16"
set title "Zero-d cell voltage solution"
set key spacing 1.2
set lmargin 9
set rmargin 2
set bmargin 3
set xlabel "time (sec)"
set ylabel "Cell voltage (V)"
plot 'zerod_analytic.dat' u 1:4 w l lw 3 title "Analytic solution",\
     'zerod_echemamr.dat' u 1:4 w p pt 6 ps 3 title "EchemAMR solution" 
