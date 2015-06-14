all:
	gcc src/latticeboltzmann.c -o bin/gcclatticeboltzmann.exe -std=gnu99 -Ofast -funroll-loops -march=core-avx-i -fopenmp -Wall -pedantic -lrt -lm

debug:
	gcc src/latticeboltzmann.c -o bin/gcclatticeboltzmann.exe -std=gnu99 -march=core-avx-i -Wall -pedantic -lrt -lm -g

icc:
	icc src/latticeboltzmann.c -o bin/icclatticeboltzmann.exe -std=gnu99 -O3 -xAVX -restrict -openmp -Wall -pedantic -lrt -lm -g

breda:
	gcc4.8.4 src/latticeboltzmann.c -o bin/gcclatticeboltzmann.exe -std=gnu99 -Ofast -funroll-loops -march=bdver2 -fopenmp -Wall -pedantic -lrt -lm

clean:
	rm bin/*.exe
