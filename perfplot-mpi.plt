set term pngcairo enhanced dashed size 1024,768 font 'Monospace'
set output "mpi-runtimes.png"


set xlabel "No. of Threads"
set ylabel "Runtime (s)"
set grid

set xrange [16:256]
set xtics 16,16,256

set yrange [0:125]
set ytics 0,25,125

set key below vertical maxrows 4



set title "Lattice-Boltzmann d2q9, 10000 timesteps"

plot "mpi-runtimes.dat" i 0 using 2:3 with linespoints lt 1 lw 2 lc rgb "brown" title "400x2000, DP, No Overlap",\
     "mpi-runtimes.dat" i 2 using 2:3 with linespoints lt 1 lw 2 lc rgb "red" title "400x2000, DP,    Overlap",\
     "mpi-runtimes.dat" i 0 using 2:5 with linespoints lt 1 lw 2 lc rgb "olive" title "400x2000, SP, No Overlap",\
     "mpi-runtimes.dat" i 2 using 2:5 with linespoints lt 1 lw 2 lc rgb "green" title "400x2000, SP,    Overlap",\
     "mpi-runtimes.dat" i 1 using 2:3 with linespoints lt 1 lw 2 lc rgb "navy" title "   800x4000, DP, No Overlap",\
     "mpi-runtimes.dat" i 3 using 2:3 with linespoints lt 1 lw 2 lc rgb "blue" title "   800x4000, DP,    Overlap",\
     "mpi-runtimes.dat" i 1 using 2:5 with linespoints lt 1 lw 2 lc rgb "#4B0082" title "   800x4000, SP, No Overlap",\
     "mpi-runtimes.dat" i 3 using 2:5 with linespoints lt 1 lw 2 lc rgb "purple" title "   800x4000, SP,    Overlap",\



set output "mpi-speedup.png"
set yrange [1:16]
set ytics 1,1,16
set ylabel "Speedup Relative to 16 Threads"
set arrow nohead from 16,1 to 256,16 lt 2 lw 2 lc rgb "black"

plot "mpi-runtimes.dat" i 0 using 2:4 with linespoints lt 1 lw 2 lc rgb "brown" title "400x2000, DP, No Overlap",\
     "mpi-runtimes.dat" i 2 using 2:4 with linespoints lt 1 lw 2 lc rgb "red" title "400x2000, DP,    Overlap",\
     "mpi-runtimes.dat" i 0 using 2:6 with linespoints lt 1 lw 2 lc rgb "olive" title "400x2000, SP, No Overlap",\
     "mpi-runtimes.dat" i 2 using 2:6 with linespoints lt 1 lw 2 lc rgb "green" title "400x2000, SP,    Overlap",\
     "mpi-runtimes.dat" i 1 using 2:4 with linespoints lt 1 lw 2 lc rgb "navy" title "   800x4000, DP, No Overlap",\
     "mpi-runtimes.dat" i 3 using 2:4 with linespoints lt 1 lw 2 lc rgb "blue" title "   800x4000, DP,    Overlap",\
     "mpi-runtimes.dat" i 1 using 2:6 with linespoints lt 1 lw 2 lc rgb "#4B0082" title "   800x4000, SP, No Overlap",\
     "mpi-runtimes.dat" i 3 using 2:6 with linespoints lt 1 lw 2 lc rgb "purple" title "   800x4000, SP,    Overlap",\

