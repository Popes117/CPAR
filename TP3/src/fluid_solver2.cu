#include "fluid_solver.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <omp.h>
#include <cuda.h>
#include <cfloat> 

#define IX(i, j, k) ((i) + (val) * (j) + (val) * (val2) * (k))  //Compute 1 dimensional (1D) index from 3D coordinates
#define SWAP(x0, x){float *tmp = x0;x0 = x;x = tmp;}            //Swap two pointers
#define MAX(a, b) (((a) > (b)) ? (a) : (b))                     //Get maximum between two values
#define LINEARSOLVERTIMES 20                                    //Number of iterations for the linear solver

//Global values to minimize the number of calculations of the index between steps
int ix000, ix100, ix010, ix001;
int ixm100, ixm00, ixm110, ixm101;
int ix0n10, ix1n10, ix0n0, ix0n11;
int ixm1n10, ixmn10, ixm1n0, ixm1n11;

__global__ void add_source_kernel(int M, int N, int O, float *x, float *s, float dt) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // O tamanho total da grade
    int size = (M + 2) * (N + 2) * (O + 2);

    // Garantir que o índice não ultrapasse o tamanho da grade
    if (idx < size) {
        x[idx] += dt * s[idx];
    }
}

void launch_add_source_kernel(int M, int N, int O, float *x, float *s, float dt) {
    int size = (M + 2) * (N + 2) * (O + 2);

    // Configuração de threads e blocos
    int threadsPerBlock = 1024;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    // Chamada ao kernel
    add_source_kernel<<<blocksPerGrid, threadsPerBlock>>>(M, N, O, x, s, dt);

}

__global__ void set_bnd_kernel(
    int M, int N, int O, int b, float *x,
    int ix010, int ix001, int ix100, int ix000,
    int ixm110, int ixm101, int ixm00, int ixm100,
    int ix0n0, int ix0n11, int ix1n10, int ix0n10,
    int ixm1n0, int ixm1n11, int ixmn10, int ixm1n10
){

    int i = threadIdx.x + blockIdx.x * blockDim.x; // thread index in x
    int j = threadIdx.y + blockIdx.y * blockDim.y; // thread index in y
    
    int val = M + 2;
    int val2 = N + 2;

    if (i >= M || j >= N || i < 0 || j < 0) return; // Ensure within bounds

    float neg_mask;

    // Handle boundaries for b == 3 (z-axis faces)
    if (b == 3) {
        neg_mask = -1.0f;
        int index = IX(0, j + 1, 0);
        int first_index = IX(0, j + 1, 1);
        int last_index = IX(0, j + 1, O);
        int idx = IX(0, j + 1, O + 1);

        if (i < M) {
            const auto first_value = x[first_index + i];
            const auto last_value = x[last_index + i];
            x[index + i] = neg_mask * first_value;
            x[idx + i] = neg_mask * last_value;
        }
    }

    // Handle boundaries for b == 1 (x-axis faces)
    if (b == 1) {
        neg_mask = -1.0f;
        int index = IX(0, j + 1, 0);
        int first_index = IX(1, j + 1, 0);
        int last_index = IX(M, j + 1, 0);
        int idx = IX(M + 1, j + 1, 0);

        if (i < M) {
            const auto first_value = x[first_index + i];
            const auto last_value = x[last_index + i];
            x[index + i] = neg_mask * first_value;
            x[idx] = neg_mask * last_value;
        }
    }

    // Handle boundaries for b == 2 (y-axis faces)
    if (b == 2) {
        neg_mask = -1.0f;
        int index = IX(i + 1, 0, 0);
        int first_index = IX(i + 1, 1, 0);
        int last_index = IX(i + 1, N, 0);
        int idx = IX(i + 1, N + 1, 0);

        if (i < M) {
            const auto first_value = x[first_index + i];
            const auto last_value = x[last_index + i];
            x[index + i] = neg_mask * first_value;
            x[idx] = neg_mask * last_value;
        }
    }

    // Handle corners (only one thread does this)
    if (i == 0 && j == 0) {
        x[ix000] = 0.33f * (x[ix100] + x[ix010] + x[ix001]);
        x[ixm100] = 0.33f * (x[ixm00] + x[ixm110] + x[ixm101]);
        x[ix0n10] = 0.33f * (x[ix1n10] + x[ix0n0] + x[ix0n11]);
        x[ixm1n10] = 0.33f * (x[ixmn10] + x[ixm1n0] + x[ixm1n11]);
    }
}

void launch_set_bnd_kernel(int M, int N, int O, int b, float *x) {

    dim3 blockDim(16, 16);
    dim3 gridDim((M + blockDim.x - 1) / blockDim.x,
                   (N + blockDim.y - 1) / blockDim.y);
    set_bnd_kernel<<<gridDim, blockDim>>>(
        M, N, O, b, x,
        ix010, ix001, ix100, ix000,
        ixm110, ixm101, ixm00, ixm100,
        ix0n0, ix0n11, ix1n10, ix0n10,
        ixm1n0, ixm1n11, ixmn10, ixm1n10
    );
   
}   

__device__ float atomicMaxFloat(float *address, float value) {
    int *address_as_int = (int *)address; // Reinterpreta o endereço como inteiro
    int old = *address_as_int, assumed;

    do {
        assumed = old;
        // Usa __int_as_float para comparar os valores como floats
        old = atomicCAS(address_as_int, assumed, 
                        __float_as_int(fmaxf(value, __int_as_float(assumed))));
    } while (assumed != old);

    return __int_as_float(old); // Retorna o valor máximo final como float
}

__device__ void atomicMaxFloatPrecise(float *address, float val) {
    float old = *address;  // Lê o valor atual
    while (val > old) {    // Continua enquanto o novo valor for maior
        float assumed = old;
        old = atomicCAS((int *)address, __float_as_int(assumed), __float_as_int(val));
    }
}

__global__ void find_max_kernel(float *arr, float *max_result, int size) {
    extern __shared__ float shared_max[];

    // Índice global e local
    unsigned int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    int tid = threadIdx.x;

    // Inicializa a memória compartilhada com o máximo de dois elementos consecutivos
    if (idx < size) {
        float val1 = arr[idx];
        float val2 = (idx + blockDim.x < size) ? arr[idx + blockDim.x] : 0.0f;
        shared_max[tid] = fmaxf(val1, val2);
    } else {
        shared_max[tid] = 0.0f;
    }

    __syncthreads();

    // Redução paralela para encontrar o máximo
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
        }
        __syncthreads();
    }

    // Redução dentro de um warp (warp-synchronous)
    if (tid < 32) {
        volatile float *vshared = shared_max; // Evita leitura de memória global
        vshared[tid] = fmaxf(vshared[tid], vshared[tid + 32]);
        vshared[tid] = fmaxf(vshared[tid], vshared[tid + 16]);
        vshared[tid] = fmaxf(vshared[tid], vshared[tid + 8]);
        vshared[tid] = fmaxf(vshared[tid], vshared[tid + 4]);
        vshared[tid] = fmaxf(vshared[tid], vshared[tid + 2]);
        vshared[tid] = fmaxf(vshared[tid], vshared[tid + 1]);
    }

    // Escreve o resultado máximo
    if (tid == 0) {
        atomicMaxFloat(max_result, shared_max[0]);
    }
}


float find_max(float *d_arr, int size) {
    int threads_per_block = 256;
    int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;

    // Memória para armazenar o resultado máximo
    float *d_max_result;
    float h_max_result = 0.0f;

    cudaMalloc(&d_max_result, sizeof(float));
    cudaMemcpy(d_max_result, &h_max_result, sizeof(float), cudaMemcpyHostToDevice);

    // Lançar o kernel
    find_max_kernel<<<blocks_per_grid, threads_per_block, threads_per_block * sizeof(float)>>>(d_arr, d_max_result, size);

    // Copiar o resultado de volta para a CPU
    cudaMemcpy(&h_max_result, d_max_result, sizeof(float), cudaMemcpyDeviceToHost);

    // Liberar memória
    cudaFree(d_max_result);

    return h_max_result;
}

template <unsigned int blockSize>
__global__ void max_reduce(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;
    unsigned int gridSize = blockSize * 2 * gridDim.x;

    // Inicializa memória compartilhada
    sdata[tid] = -FLT_MAX;

    // Carrega elementos para memória compartilhada
    while (i < n) {
        sdata[tid] = fmaxf(sdata[tid], g_idata[i]);
        if (i + blockSize < n) {  // Garante que não acesse fora do limite
            sdata[tid] = fmaxf(sdata[tid], g_idata[i + blockSize]);
        }
        i += gridSize;
    }

    //float temp_max = -FLT_MAX;
    //while (i < n) {
    //    temp_max = fmaxf(temp_max, g_idata[i]);
    //    if (i + blockSize < n) {  
    //        temp_max = fmaxf(temp_max, g_idata[i + blockSize]);
    //    }
    //    i += gridSize;
    //}
    //sdata[tid] = temp_max;

    __syncthreads();

    // Redução em memória compartilhada
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] = fmaxf(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] = fmaxf(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64)  { sdata[tid] = fmaxf(sdata[tid], sdata[tid + 64]);  } __syncthreads(); }

    if (tid < 32) {
        volatile float *vshared = sdata;
        vshared[tid] = fmaxf(vshared[tid], vshared[tid + 32]);
        vshared[tid] = fmaxf(vshared[tid], vshared[tid + 16]);
        vshared[tid] = fmaxf(vshared[tid], vshared[tid + 8]);
        vshared[tid] = fmaxf(vshared[tid], vshared[tid + 4]);
        vshared[tid] = fmaxf(vshared[tid], vshared[tid + 2]);
        vshared[tid] = fmaxf(vshared[tid], vshared[tid + 1]);
    }

    // Escreve o resultado do bloco na memória global
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

void launch_max_reduce(float *d_idata, float *d_odata, unsigned int n) {
    const unsigned int blockSize = 512; // Pode ajustar para diferentes GPUs
    const unsigned int gridSize = (n + blockSize * 2 - 1) / (blockSize * 2);

    // Aloca memória para a saída da redução
    float *d_intermediate;
    cudaMalloc(&d_intermediate, gridSize * sizeof(float));

    // Lança o kernel
    max_reduce<blockSize><<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_idata, d_intermediate, n);

    // Reduz o resultado dos blocos em um único valor
    max_reduce<blockSize><<<1, blockSize, blockSize * sizeof(float)>>>(d_intermediate, d_odata, gridSize);

    cudaFree(d_intermediate);
}

float find_max2(float *changes_d, int n){

    float max_value;
    int num_blocks = (n + 511) / 512;
    float *d_intermediate;

    cudaMalloc(&d_intermediate, num_blocks * sizeof(float));

    max_reduce<512><<<num_blocks, 512, 512 * sizeof(float)>>>(changes_d, d_intermediate, n);
    // Reduzir os resultados intermediários até que reste apenas um bloco
    while(num_blocks > 1){
        int num_blocks_next = (num_blocks + 511) / 512;
        max_reduce<512><<<num_blocks_next, 512, 512 * sizeof(float)>>>(d_intermediate, d_intermediate, num_blocks);
        num_blocks = num_blocks_next;
    }

    cudaMemcpy(&max_value, d_intermediate, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_intermediate);

    return max_value;

}

__global__ void lin_solve_kernel(int M, int N, int O, int b, float *x, float *x0, float a, float c, bool process_red, float *changes) {
    
    int val = M + 2;
    int val2 = N + 2;
    float divv = 1.0f / c;
    int y = M + 2;
    int z = (M + 2) * (N + 2);
    int color = int(process_red);

    // Índices globais baseados em thread e bloco
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int i = 2 * (blockIdx.x * blockDim.x + threadIdx.x) + 1 + (j + k + color) % 2 ; // Garante que começa em 1

    // Verifica se está dentro dos limites
    if (i > M || j > N || k > O) return;

    int idx = IX(i, j, k);
    float old_x = x[idx];

    // Atualiza o valor de x[idx] com a fórmula dada
    x[idx] = (x0[idx] +
              a * (x[idx - 1] + x[idx + 1] +
                   x[idx - y] + x[idx + y] +
                   x[idx - z] + x[idx + z])) * divv;

    // Calcula a alteração e atualiza max_c de forma atômica
    changes[idx] = fabsf(x[idx] - old_x);
}


void lin_solve_kernel(int M, int N, int O, int b, float *x, float *x0, float a, float c) {
    float tol = 1e-7, max_c;
    int l = 0;
    int size = (M + 2) * (N + 2) * (O + 2) * sizeof(float);
    float *changes_d, *d_max_c;

    cudaMalloc(&changes_d, size);
    cudaMalloc(&d_max_c, sizeof(float));

    // Configuração do grid e blocos
    dim3 blockDim(16, 16, 4);
    dim3 gridDim((M/2 + blockDim.x - 1) / blockDim.x,
                 (N + blockDim.y - 1) / blockDim.y,
                 (O + blockDim.z - 1) / blockDim.z);

    do {
        max_c = 0.0f;
        cudaMemset(changes_d, 0, size);

        // Processa células pretas
        lin_solve_kernel<<<gridDim, blockDim>>>(M, N, O, b, x, x0, a, c, false, changes_d);

        // Processa células vermelhas
        lin_solve_kernel<<<gridDim, blockDim>>>(M, N, O, b, x, x0, a, c, true, changes_d);

        // Calcula o valor máximo em `changes_d`
        launch_max_reduce(changes_d, d_max_c, (M + 2) * (N + 2) * (O + 2));

        // Copia o resultado para o host
        cudaMemcpy(&max_c, d_max_c, sizeof(float), cudaMemcpyDeviceToHost);

        // Atualiza os limites com o kernel `set_bnd`
        launch_set_bnd_kernel(M, N, O, b, x);

    } while (max_c > tol && ++l < 20);

    // Libera a memória alocada para `d_max_c`
    cudaFree(changes_d);
}

// Diffusion step (uses implicit method)
void diffuse(int M, int N, int O, int b, float *x, float *x0, float diff, float dt) {
    
    int max = MAX(MAX(M, N), O);
    float a = dt * diff * max * max;
    lin_solve_kernel(M, N, O, b, x, x0, a, 1 + 6 * a);

}

__global__ void advect_kernel(int M, int N, int O, int b, float *d, float *d0, float *u, float *v, float *w, float dt) {

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

}

void launch_advect_kernel(int M, int N, int O, int b, float *d, float *d0, float *u, float *v, float *w, float dt) {

    dim3 blockDim(16, 16, 4);
    dim3 gridDim((M + 2 + blockDim.x - 1) / blockDim.x,
                   (N + 2 + blockDim.y - 1) / blockDim.y,
                   (O + 2 + blockDim.z - 1) / blockDim.z);
    advect_kernel<<<gridDim,blockDim>>>(M, N, O, b, d, d0, u, v, w, dt);
    launch_set_bnd_kernel(M, N, O, b, d);
    
}   


// Advection step (uses velocity field to move quantities)
void advect(int M, int N, int O, int b, float *d, float *d0, float *u, float *v, float *w, float dt) {
    float dtX = dt * M, dtY = dt * N, dtZ = dt * O;

    launch_advect_kernel(M, N, O, b, d, d0, u, v, w, dt);  
    launch_set_bnd_kernel(M, N, O, b, d);

}

__global__ void loop1_project_kernel(int M, int N, int O, float *u, float *v, float *w, float *p, float *div) {
    int val = M + 2;
    int val2 = N + 2;
    int y = M + 2;
    int z = (M + 2) * (N + 2); 
    int max = MAX(M, MAX(N, O));
    float invMax = 1.0f / max;

    int i = blockIdx.x * blockDim.x + threadIdx.x + 1; // Índice em x
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1; // Índice em y
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1; // Índice em z

    if (i < 1 || j < 1 || k < 1 || i > M || j > N || k > O) return;

    int idx = IX(i, j, k);

    div[idx] = (-0.5f * (u[idx + 1] - u[idx - 1] + v[idx + y] -
                         v[idx - y] + w[idx + z] - w[idx - z])) * invMax;
    p[idx] = 0.0f;
}

__global__ void loop2_project_kernel(int M, int N, int O, float *u, float *v, float *w, float *p) {
    int val = M + 2;
    int val2 = N + 2;
    int y = M + 2;
    int z = (M + 2) * (N + 2); 
    int max = MAX(M, MAX(N, O));
    float invMax = 1.0f / max;

    int i = blockIdx.x * blockDim.x + threadIdx.x + 1; // Índice em x
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1; // Índice em y
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1; // Índice em z

    // Adjustment of u, v, and w without loop blocking

    if (i < 1 || j < 1 || k < 1 || i > M || j > N || k > O) return;

    int idx = IX(i, j, k);

    u[idx] -= 0.5f * (p[idx + 1] - p[idx - 1]);
    v[idx] -= 0.5f * (p[idx + y] - p[idx - y]);
    w[idx] -= 0.5f * (p[idx + z] - p[idx - z]);
}

void project(int M, int N, int O, float *u, float *v, float *w, float *p, float *divv) {
    int max = MAX(M, MAX(N, O));
    float invMax = 1.0f / max;

    dim3 blockDim(16, 16, 4);
    dim3 gridDim((M + blockDim.x - 1) / blockDim.x,
                   (N + blockDim.y - 1) / blockDim.y,
                   (O + blockDim.z - 1) / blockDim.z);
    loop1_project_kernel<<<gridDim, blockDim>>>(M, N, O, u, v, w, p, divv);
    cudaDeviceSynchronize();

    launch_set_bnd_kernel(M, N, O, 0, divv);
    launch_set_bnd_kernel(M, N, O, 0, p);
    lin_solve_kernel(M, N, O, 0, p, divv, 1, 6);
    //cudaDeviceSynchronize();

    loop2_project_kernel<<<gridDim, blockDim>>>(M, N, O, u, v, w, p);
    cudaDeviceSynchronize();

    launch_set_bnd_kernel(M, N, O, 1, u),
    launch_set_bnd_kernel(M, N, O, 2, v);
    launch_set_bnd_kernel(M, N, O, 3, w);

}

// Step function for density
void dens_step(int M, int N, int O, float *x, float *x0, float *u, float *v, float *w, float diff, float dt) {  

  launch_add_source_kernel(M, N, O, x, x0, dt); 
  SWAP(x0, x);
  diffuse(M, N, O, 0, x, x0, diff, dt);
  SWAP(x0, x);
  advect(M, N, O, 0, x, x0, u, v, w, dt);

}

// Step function for velocity
void vel_step(int M, int N, int O, float *u, float *v, float *w, float *u0, float *v0, float *w0, float visc, float dt) {
  // Define global values
  int val = M + 2;
  int val2 = N + 2;
  ix000 = IX(0, 0, 0);
  ix100 = IX(1, 0, 0);
  ix010 = IX(0, 1, 0);
  ix001 = IX(0, 0, 1);
  ixm100 = IX(M + 1, 0, 0);
  ixm00 = IX(M, 0, 0);
  ixm110 = IX(M + 1, 1, 0);
  ixm101 = IX(M + 1, 0, 1);
  ix0n10 = IX(0, N + 1, 0);
  ix1n10 = IX(1, N + 1, 0);
  ix0n0 = IX(0, N, 0);
  ix0n11 = IX(0, N + 1, 1);
  ixm1n10 = IX(M + 1, N, 0);
  ixmn10 = IX(M, N + 1, 0);
  ixm1n0 = IX(M + 1, N, 0);
  ixm1n11 = IX(M + 1, N + 1, 1);

  launch_add_source_kernel(M, N, O, u, u0, dt);
  launch_add_source_kernel(M, N, O, v, v0, dt);
  launch_add_source_kernel(M, N, O, w, w0, dt);

  SWAP(u0, u);
  diffuse(M, N, O, 1, u, u0, visc, dt);
  SWAP(v0, v);
  diffuse(M, N, O, 2, v, v0, visc, dt);
  SWAP(w0, w);
  diffuse(M, N, O, 3, w, w0, visc, dt);
  project(M, N, O, u, v, w, u0, v0);
  SWAP(u0, u);
  SWAP(v0, v);
  SWAP(w0, w);
  advect(M, N, O, 1, u, u0, u0, v0, w0, dt);
  advect(M, N, O, 2, v, v0, u0, v0, w0, dt);
  advect(M, N, O, 3, w, w0, u0, v0, w0, dt);
  project(M, N, O, u, v, w, u0, v0);
}
