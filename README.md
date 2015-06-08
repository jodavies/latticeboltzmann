# latticeboltzmann

Implementation of D2Q9 Lattice Boltzmann, with Bhatnagar-Gross-Krook collision approximation.

![Flow](img/flow.gif)

Space is discretized on a 2d lattice, and particle distribution functions ("velocities") are discretized onto the links between lattice points and give the probability that a particle will move from one lattice point to the next, denoted:

```
f6  f2  f5
  \  |  /
f3--f0--f1
  /  |  \
f7  f4  f8
```

Lattice-site local macroscopic variables `rho = Sum f_i` and velocity `u = 1/rho Sum f_i e_i`, where `e_i` are vectors pointing along the lattice links.

Each timestep, the particle distribution functions are updated as

`f_i(x+e_i*dt, t+dt) = f_i(x,t) - 1/tau [f_i(x,t) - feq_i(x,t)]`

where (in the BGK approximation)

`feq_i = omega_i rho(x) [1 + 3/2 e_i.u(x)/c^2 + 9/2 (e_i.u(x))^2/c^4 - 3/2 u(x)^2/c^2]`.

At walls we implement "bounce back" (no slip) conditions, and the domain wraps around at the edges.


### CPU Implementation
Implemented with AVX/SSE vector intrinsics and OpenMP. We vectorize over the "long" y direction, and parallelize over the shorter x direction. The algorithm is very memory-bandwidth constrained, since the computation scales linearly with the lattice size.

As the plot below shows, for some CPUs we receive no benefit whatsoever by parallelizing across multiple cores with OpenMP.

![Runtime](img/runtimes.png)

Plotting the speedup relative to a single thread alongside the bandwidth improvement factor by adding threads to the STREAM Triad benchmark, we see that parallel speedup here is only possible if adding threads provides access to additional memory bandwidth. We see also the sensitivity to running scheduling threads appropriately on NUMA machines.

![Speedup](img/speedup.png)


### OpenCL
The NVIDIA GTX960 available for testing as around 10x the memory bandwidth available compared to the i5-2500K system, so we can expect a similar speedup by porting the code to OpenCL.