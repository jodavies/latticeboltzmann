all:
	which mpicc
	mpicc src/latticeboltzmann.c -o bin/gcclatticeboltzmann.exe -std=gnu99 -Ofast -funroll-loops -march=core-avx-i -Wall -pedantic -lrt -lm

debug:
	which mpicc
	mpicc src/latticeboltzmann.c -o bin/gcclatticeboltzmann.exe -std=gnu99 -march=core-avx-i -Wall -pedantic -lrt -lm -g

ulg:
	which mpicc
	mpicc src/latticeboltzmann.c -o bin/gcclatticeboltzmann.exe -std=gnu99 -Ofast -funroll-loops -march=barcelona -Wall -pedantic -lrt -lm -lmpi

ulgvt:
	mpicc-vt -vt:cc mpicc src/latticeboltzmann.c -o bin/vtgcclatticeboltzmann.exe -std=gnu99 -Ofast -funroll-loops -march=barcelona -Wall -pedantic -lrt -lm -lmpi -g


clean:
	rm bin/*.exe
