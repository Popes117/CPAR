#include "fluid_solver.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <omp.h>
#include <cuda.h>

#define IX(i, j, k) ((i) + (val) * (j) + (val) * (val2) * (k))  //Compute 1 dimensional (1D) index from 3D coordinates
#define SWAP(x0, x){float *tmp = x0;x0 = x;x = tmp;}            //Swap two pointers
#define MAX(a, b) (((a) > (b)) ? (a) : (b))                     //Get maximum between two values
#define LINEARSOLVERTIMES 20                                    //Number of iterations for the linear solver

#define NUM_BLOCKS 512
#define NUM_THREADS_PER_BLOCK 256
#define TOTALSIZE NUM_BLOCKS*NUM_THREADS_PER_BLOCK

//Global values to minimize the number of calculations of the index between steps
int ix000, ix100, ix010, ix001;
int ixm100, ixm00, ixm110, ixm101;
int ix0n10, ix1n10, ix0n0, ix0n11;
int ixm1n10, ixmn10, ixm1n0, ixm1n11;

// Clamp value between a given minimum and maximum
inline float clamp(float value, float minVal, float maxVal) {
    return std::max(minVal, std::min(value, maxVal));
}


// Diffusion step (uses implicit method)
__global__ void diffuse_kernel(int M, int N, int O, int b, float *x, float *x0, float diff, float dt) {
    int max = MAX(MAX(M, N), O);
    int size = (M + 2) * (N + 2) * (O + 2) * sizeof(float);
    float a = dt * diff * max * max;
    float *x_d, *x0_d;

    cudaMalloc((void **)&x_d, size);
    cudaMemcpy(x_d, x, size, cudaMemcpyHostToDevice);   

    dim3 blockDim(16, 16, 4);
    dim3 gridDim((M + 2 + blockDim.x - 1) / blockDim.x,
                   (N + 2 + blockDim.y - 1) / blockDim.y,
                   (O + 2 + blockDim.z - 1) / blockDim.z);
    lin_solve_kernel<<<gridDim,blockDim>>>(M, N, O, b, x, x0, a, 1 + 6 * a);

    cudaMemcpy(x, x_d, size, cudaMemcpyDeviceToHost);   
    cudaFree(x_d);    
}

__global__ void advect_kernel(int M, int N, int O, int b, float *d, const float *d0, float *u, float *v, float *w, float dt) {

    float dtX = dt * M, dtY = dt * N, dtZ = dt * O;
    int val = M + 2;
    int val2 = N + 2;

    int i = blockIdx.x * blockDim.x + threadIdx.x + 1; // Índice em x
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1; // Índice em y
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1; // Índice em z

    if (i > M || j > N || k > O) return; // Fora dos limites

    int idx = IX(i, j, k);

    // Calcula as posições retroativas
    float x = i - dtX * u[idx];
    float y = j - dtY * v[idx];
    float z = k - dtZ * w[idx];

    // Clamping para garantir que esteja dentro do domínio
    x = fminf(fmaxf(x, 0.5f), M + 0.5f);
    y = fminf(fmaxf(y, 0.5f), N + 0.5f);
    z = fminf(fmaxf(z, 0.5f), O + 0.5f);

    // Índices inteiros para interpolação
    int i0 = (int)x, i1 = i0 + 1;
    int j0 = (int)y, j1 = j0 + 1;
    int k0 = (int)z, k1 = k0 + 1;

    // Pesos de interpolação
    float s1 = x - i0, s0 = 1 - s1;
    float t1 = y - j0, t0 = 1 - t1;
    float u1 = z - k0, u0 = 1 - u1;

    // Recuperar valores para interpolação
    float d0_i0j0k0 = d0[IX(i0, j0, k0)];
    float d0_i0j0k1 = d0[IX(i0, j0, k1)];
    float d0_i0j1k0 = d0[IX(i0, j1, k0)];
    float d0_i0j1k1 = d0[IX(i0, j1, k1)];
    float d0_i1j0k0 = d0[IX(i1, j0, k0)];
    float d0_i1j0k1 = d0[IX(i1, j0, k1)];
    float d0_i1j1k0 = d0[IX(i1, j1, k0)];
    float d0_i1j1k1 = d0[IX(i1, j1, k1)];

    // Interpolação 3D
    d[idx] = s0 * (t0 * (u0 * d0_i0j0k0 + u1 * d0_i0j0k1) +
                   t1 * (u0 * d0_i0j1k0 + u1 * d0_i0j1k1)) +
             s1 * (t0 * (u0 * d0_i1j0k0 + u1 * d0_i1j0k1) +
                   t1 * (u0 * d0_i1j1k0 + u1 * d0_i1j1k1));

    launch_set_bnd_kernel(M, N, O, b, d);
}

void launch_advect_kernel(int M, int N, int O, int b, float *d, const float *d0, float *u, float *v, float *w, float dt) {

    float *d_d, *d0_d, *u_d, *v_d, *w_d;
    int size = (M + 2) * (N + 2) * (O + 2) * sizeof(float);

    cudaMalloc((void **)&d_d, size);
    cudaMalloc((void **)&d0_d, size);
    cudaMalloc((void **)&u_d, size);
    cudaMalloc((void **)&v_d, size);
    cudaMalloc((void **)&w_d, size);

    cudaMemcpy(d_d, d, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d0_d, d0, size, cudaMemcpyHostToDevice);
    cudaMemcpy(u_d, u, size, cudaMemcpyHostToDevice);
    cudaMemcpy(v_d, v, size, cudaMemcpyHostToDevice);
    cudaMemcpy(w_d, w, size, cudaMemcpyHostToDevice);   

    dim3 blockDim(16, 16, 4);
    dim3 gridDim((M + 2 + blockDim.x - 1) / blockDim.x,
                   (N + 2 + blockDim.y - 1) / blockDim.y,
                   (O + 2 + blockDim.z - 1) / blockDim.z);
    advect_kernel<<<gridDim,blockDim>>>(M, N, O, b, d_d, d0_d, u_d, v_d, w_d, dt);

    cudaMemcpy(d, d_d, size, cudaMemcpyDeviceToHost);  

    cudaFree(d_d);
    cudaFree(d0_d);
    cudaFree(u_d);
    cudaFree(v_d);
    cudaFree(w_d);    
}   