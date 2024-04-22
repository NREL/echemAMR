set terminal png
set output "zerodsoln_conc_1d.png"
set termoption font "Helvetica,16"
set title "Zero-d conc. solution"
set key spacing 1.2
set lmargin 9
set rmargin 2
set bmargin 3
set xlabel "time (sec)"
set ylabel "Conc. (mol/m3)"
set key center
plot 'zerod_analytic.dat' u 1:2 w l lw 3 title "Analytic solution (anode)",\
     'zerod_analytic.dat' u 1:3 w l lw 3 title "Analytic solution (cathode)",\
     'zerod_echemamr.dat' u 1:2 w p ps 2 pt 6 title "EchemAMR solution (anode)",\
     'zerod_echemamr.dat' u 1:3 w p ps 2 pt 6 title "EchemAMR solution (cathode)" 
