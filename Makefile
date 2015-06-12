all:
	gcc src/latticeboltzmann.c -o bin/gcclatticeboltzmann.exe -std=gnu99 -Ofast -Wall -pedantic -lrt -lm -lOpenCL -g

pocl:
	gcc src/latticeboltzmann.c -o bin/gcclatticeboltzmann.exe -std=gnu99 -Ofast -Wall -pedantic -lrt -lm -lpocl -g

clean:
	rm bin/*.exe
