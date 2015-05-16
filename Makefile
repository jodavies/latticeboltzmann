all:
	gcc src/latticeboltzmann.c -o bin/gcclatticeboltzmann.exe -std=gnu99 -Ofast -funroll-loops -march=core-avx-i -Wall -pedantic -lrt -lm -g

icc:
	icc src/latticeboltzmann.c -o bin/icclatticeboltzmann.exe -std=gnu99 -O3 -xAVX -restrict -Wall -pedantic -lrt -lm -g

clean:
	rm bin/*.exe
