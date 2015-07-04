all:
	gcc src/latticeboltzmann.c -o bin/gcclatticeboltzmann.exe -std=gnu99 -Ofast -fopenmp -Wall -pedantic -lrt -lm -lOpenCL

pocl:
	gcc src/latticeboltzmann.c -o bin/gcclatticeboltzmann.exe -std=gnu99 -Ofast -fopenmp -Wall -pedantic -lrt -lm -lpocl -g

clean:
	rm bin/*.exe
