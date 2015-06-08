set term pngcairo enhanced dashed size 1024,768 font 'Monospace'
set output "runtimes.png"


set xlabel "No. of Threads"
set ylabel "Runtime (s)"
set grid

set logscale x 2
set xrange [1:16]
set xtics (1,2,4,8,16)

set yrange [0:950]

set key below



set title "Lattice-Boltzmann d2q9, 400x2000 Lattice, 10000 timesteps"

plot "runtimes.dat" i 0 using 1:3 with linespoints lt 1 lw 2 lc rgb "red" title "Core i5-2500K AVX",\
     "runtimes.dat" i 1 using 1:3 with linespoints lt 1 lw 2 lc rgb "navy" title "2x Xeon E5-2667v2 One NUMA Node  AVX",\
     "runtimes.dat" i 2 using 1:3 with linespoints lt 1 lw 2 lc rgb "blue" title "2x Xeon E5-2667v2 Two NUMA Nodes AVX",\
     "runtimes.dat" i 3 using 1:3 with linespoints lt 1 lw 2 lc rgb "olive" title "2x Opteron 6128 One  NUMA Node  SSE",\
     "runtimes.dat" i 4 using 1:3 with linespoints lt 1 lw 2 lc rgb "green" title "2x Opteron 6128 Four NUMA Nodes SSE",\
     "runtimes.dat" i 5 using 1:3 with linespoints lt 1 lw 2 lc rgb "#4B0082" title "2x Opteron 6328 One  NUMA Node  AVX",\
     "runtimes.dat" i 6 using 1:3 with linespoints lt 1 lw 2 lc rgb "purple" title "2x Opteron 6328 Four NUMA Nodes AVX",\



set output "speedup.png"
set yrange [0:16]
set ylabel "Speedup Relative to 1 Thread"

plot "runtimes.dat" i 0 using 1:4 with linespoints lt 1 lw 2 lc rgb "red" title "Core i5-2500K AVX  ",\
     "runtimes.dat" i 0 using 1:2 with linespoints lt 3 lw 2 lc rgb "red" title "Core i5-2500K Triad",\
     "runtimes.dat" i 1 using 1:4 with linespoints lt 1 lw 2 lc rgb "navy" title "2x Xeon E5-2667v2 One NUMA Node  AVX  ",\
     "runtimes.dat" i 1 using 1:2 with linespoints lt 3 lw 2 lc rgb "navy" title "2x Xeon E5-2667v2 One NUMA Node  Triad",\
     "runtimes.dat" i 2 using 1:4 with linespoints lt 1 lw 2 lc rgb "blue" title "2x Xeon E5-2667v2 Two NUMA Nodes AVX  ",\
     "runtimes.dat" i 2 using 1:2 with linespoints lt 3 lw 2 lc rgb "blue" title "2x Xeon E5-2667v2 Two NUMA Nodes Triad",\
     "runtimes.dat" i 3 using 1:4 with linespoints lt 1 lw 2 lc rgb "olive" title "2x Opteron 6128 One  NUMA Node  SSE  ",\
     "runtimes.dat" i 3 using 1:2 with linespoints lt 3 lw 2 lc rgb "olive" title "2x Opteron 6128 One  NUMA Node  Triad",\
     "runtimes.dat" i 4 using 1:4 with linespoints lt 1 lw 2 lc rgb "green" title "2x Opteron 6128 Four NUMA Nodes SSE  ",\
     "runtimes.dat" i 4 using 1:2 with linespoints lt 3 lw 2 lc rgb "green" title "2x Opteron 6128 Four NUMA Nodes Triad",\
     "runtimes.dat" i 5 using 1:4 with linespoints lt 1 lw 2 lc rgb "#4B0082" title "2x Opteron 6328 One  NUMA Node  AVX  ",\
     "runtimes.dat" i 5 using 1:2 with linespoints lt 3 lw 2 lc rgb "#4B0082" title "2x Opteron 6328 One  NUMA Node  Triad",\
     "runtimes.dat" i 6 using 1:4 with linespoints lt 1 lw 2 lc rgb "purple" title "2x Opteron 6328 Four NUMA Nodes AVX  ",\
     "runtimes.dat" i 6 using 1:2 with linespoints lt 3 lw 2 lc rgb "purple" title "2x Opteron 6328 Four NUMA Nodes Triad",\
