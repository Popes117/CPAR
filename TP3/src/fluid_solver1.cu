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

// Add sources (density or velocity)
void add_source(int M, int N, int O, float *x, float *s, float dt) {
  int size = (M + 2) * (N + 2) * (O + 2);
  for (int i = 0; i < size; i++) {
    x[i] += dt * s[i];
  }
}

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

    // Sincronizar para garantir que o kernel tenha concluído a execução
    cudaDeviceSynchronize();
}


// Set boundary conditions
void set_bnd(int M, int N, int O, int b, float *x) {
  int i, j;
  int val = M + 2;
  int val2 = N + 2;
  int y = M + 2;
  auto neg_mask = (b == 3) ? -1.0F : 1.0F;
  auto index = IX(0, 1, 0);
  auto first_index = IX(0, 1, 1);
  auto last_index = IX(0, 1, O);
  int idx = IX(0,1,O + 1);
  
  // Set boundary on faces
  for (j = 1; j <= N; j++) {
    for (i = 1; i <= M; i++) {
      const auto first_value = x[first_index + i];
      const auto last_value = x[last_index + i];
      x[index + i] = neg_mask * first_value;
      x[idx + i] = neg_mask * last_value;
    }
    index += y;
    first_index += y;
    last_index += y;
    idx += y;
  }

  // Mask for b == 1 in the second loop (x-axis)
  neg_mask = (b == 1) ? -1.0F : 1.0F;
  index = IX(0, 1, 0);
  first_index = IX(1, 1, 0);
  last_index = IX(M, 1, 0);
  idx = IX(M + 1, 1, 0);

  // Set boundaries on the x faces
  for (j = 1; j <= N; j++) {
      for (i = 1; i <= M; i++) {
          const auto first_value = x[first_index + i];
          const auto last_value = x[last_index + i];
          x[index + i] = neg_mask * first_value;
          x[idx] = neg_mask * last_value;
      }
      index += y;
      first_index += y;
      last_index += y;
      idx += y;
  }

  // Mask for b == 2 in the third loop (y-axis)
  neg_mask = (b == 2) ? -1.0F : 1.0F;
  index = IX(1, 0, 0);
  first_index = IX(1, 1, 0);
  last_index = IX(1, N, 0);
  idx = IX(1, N + 1, 0);

  // Set boundaries on the y faces
  for (j = 1; j <= N; j++) {
      const auto first_value = x[first_index + j];
      const auto last_value = x[last_index + j];
      x[index] = neg_mask * first_value;
      x[idx] = neg_mask * last_value;
      index += 1;
      first_index += 1;
      last_index += 1;
      idx += 1;
  }

  // Set corners
  x[ix000] = 0.33f * (x[ix100] + x[ix010] + x[ix001]);
  x[ixm100] = 0.33f * (x[ixm00] + x[ixm110] + x[ixm101]);
  x[ix0n10] = 0.33f * (x[ix1n10] + x[ix0n0] + x[ix0n11]);
  x[ixm1n10] = 0.33f * (x[ixmn10] + x[ixm1n0] + x[ixm1n11]);

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

    int size = (M + 2) * (N + 2) * (O + 2) * sizeof(float);  

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

__global__ void lin_solve_kernel(int M, int N, int O, int b, float *x, float *x0, float a, float c, float *max_c, bool process_red) {
    
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

    // Determina o tipo de célula a ser processada com base na paridade de (i + j + k)
    //int cell_type = (i + j + k) % 2; 
    //if ((process_red && cell_type != 0) || (!process_red && cell_type != 1)) return;

    int idx = IX(i, j, k);
    float old_x = x[idx];

    // Atualiza o valor de x[idx] com a fórmula dada
    x[idx] = (x0[idx] +
              a * (x[idx - 1] + x[idx + 1] +
                   x[idx - y] + x[idx + y] +
                   x[idx - z] + x[idx + z])) * divv;

    // Calcula a alteração e atualiza max_c de forma atômica
    float change = fabsf(x[idx] - old_x);
    atomicMaxFloat(max_c, change); // Atualiza max_c usando operação atômica
}


void lin_solve_kernel(int M, int N, int O, int b, float *x, float *x0, float a, float c) {
    float tol = 1e-7, max_c;
    int l = 0;

    // Aloca `max_c` em memória global no device
    float *d_max_c;
    cudaMalloc(&d_max_c, sizeof(float));

    // Configuração do grid e blocos
    dim3 blockDim(16, 16, 4);
    // Para a maneira do Artur, divide o M/2
    dim3 gridDim((M/2 + blockDim.x - 1) / blockDim.x,
                 (N + blockDim.y - 1) / blockDim.y,
                 (O + blockDim.z - 1) / blockDim.z);

    do {
        max_c = 0.0f;
        
        // Inicializa `max_c` para 0.0f no host e copia para o device
        cudaMemcpy(d_max_c, &max_c, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_max_c, 0, sizeof(float)); // Inicializa `d_max_c` para 0 no dispositivo

        // Executa o kernel para células vermelhas
        //std::cout << max_c << std::endl;
        //red_lin_solve_kernel<<<gridDim, blockDim>>>(M, N, O, b, x_d, x0_d, a, c, d_max_c);
        //cudaDeviceSynchronize();

        //std::cout << "Red done" << std::endl;

        // Executa o kernel para células pretas
        //black_lin_solve_kernel<<<gridDim, blockDim>>>(M, N, O, b, x_d, x0_d, a, c, d_max_c);
        //cudaDeviceSynchronize();

        // Processa células pretas
        lin_solve_kernel<<<gridDim, blockDim>>>(M, N, O, b, x, x0, a, c, d_max_c, false);
        cudaDeviceSynchronize();

        // Processa células vermelhas
        lin_solve_kernel<<<gridDim, blockDim>>>(M, N, O, b, x, x0, a, c, d_max_c, true);
        cudaDeviceSynchronize();

        // Copia `max_c` de volta para o host para verificação
        cudaMemcpy(&max_c, d_max_c, sizeof(float), cudaMemcpyDeviceToHost);
        //std::cout << "max_c (host) antes da set_bnd : " << max_c << std::endl;
        // Atualiza os limites com o kernel `set_bnd`
        launch_set_bnd_kernel(M, N, O, b, x);

    } while (++l < 20);

    // Libera a memória alocada para `d_max_c`
    cudaFree(d_max_c);
}

// red-black solver with convergence check
void lin_solve(int M, int N, int O, int b, float *x, float *x0, float a, float c) {
    float tol = 1e-7, max_c;
    int l = 0;
    int val = M + 2;
    int val2 = N + 2;
    float div = 1/c;
    int y = M + 2;
    int z = (M + 2) * (N + 2);
    float *x_d;
    
    do {
        max_c = 0.0f;
        for (int k = 1; k <= O; k++) {
            for (int j = 1; j <= N; j++) {
                for (int i = 1 + (j + k) % 2; i <= M; i += 2) {
                    int idx = IX(i, j, k);
                    float old_x = x[idx];
                    x[idx] = (x0[idx] +
                              a * (x[idx - 1] + x[idx + 1] +
                                   x[idx - y] + x[idx + y] +
                                   x[idx - z] + x[idx + z])) * div;
                    float change = fabs(x[idx] - old_x);
                    if (change > max_c) max_c = change;
                }
            }
        }

        for (int k = 1; k <= O; k++) {
            for (int j = 1; j <= N; j++) {
                for (int i = 1 + (j + k + 1) % 2; i <= M; i += 2) {
                    int idx = IX(i, j, k);
                    float old_x = x[idx];
                    x[idx] = (x0[idx] +
                              a * (x[idx - 1] + x[idx + 1] +
                                   x[idx - y] + x[idx + y] +
                                   x[idx - z] + x[idx + z])) * div;
                    float change = fabs(x[idx] - old_x);
                    if (change > max_c) max_c = change;
                }
            }
        }

        set_bnd(M, N, O, b, x);

        //cudaMalloc((void **)&x_d, (M + 2) * (N + 2) * (O + 2) * sizeof(float));
        //cudaMemcpy(x_d, x, (M + 2) * (N + 2) * (O + 2) * sizeof(float), cudaMemcpyHostToDevice);
        //launch_set_bnd_kernel(M, N, O, b, x_d);
        //cudaMemcpy(x, x_d, (M + 2) * (N + 2) * (O + 2) * sizeof(float), cudaMemcpyDeviceToHost);
        //cudaFree(x_d); 
    } while (max_c > tol && ++l < 20);
}

// Diffusion step (uses implicit method)
void diffuse(int M, int N, int O, int b, float *x, float *x0, float diff, float dt) {
    int max = MAX(MAX(M, N), O);
    float a = dt * diff * max * max;
    float *x_d, *x0_d;
    int size = (M + 2) * (N + 2) * (O + 2) * sizeof(float);
    float *copy = (float *)malloc(size);
    memcpy(copy, x, size); 
    //lin_solve(M, N, O, b, x, x0, a, 1 + 6 * a);

    cudaMalloc((void **)&x_d, size);
    cudaMalloc((void **)&x0_d, size);
    cudaMemcpy(x_d, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(x0_d, x0, size, cudaMemcpyHostToDevice);

    lin_solve_kernel(M, N, O, b, x_d, x0_d, a, 1 + 6 * a);
    cudaDeviceSynchronize();

    cudaMemcpy(x, x_d, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    cudaFree(x_d);
    cudaFree(x0_d);

    bool is_different = false;
    for (int idx = 0; idx < (M + 2) * (N + 2) * (O + 2); idx++) {
        if (x[idx] != copy[idx]) {
            is_different = true;
            break;
        }
    }

    // Print the result
    if (is_different){
        printf("x is different from the copy after lin_solve.\n");
    }
    else {
        printf("x is identical to the copy after lin_solve.\n");
    }
}

// Advection step (uses velocity field to move quantities)
void advect(int M, int N, int O, int b, float *d, float *d0, float *u, float *v, float *w, float dt) {
    float dtX = dt * M, dtY = dt * N, dtZ = dt * O;
    int val = M + 2;
    int val2 = N + 2;
    float *d_d;
    int size = (M + 2) * (N + 2) * (O + 2) * sizeof(float);

    for (int k = 1; k <= O; k++) {
        for (int j = 1; j <= N; j++) {
            for (int i = 1; i <= M; i++) {
                int idx = IX(i, j, k);

                float x = i - dtX * u[idx];
                float y = j - dtY * v[idx];
                float z = k - dtZ * w[idx];

                // Optimized clamping using inline function
                x = clamp(x, 0.5f, M + 0.5f);
                y = clamp(y, 0.5f, N + 0.5f);
                z = clamp(z, 0.5f, O + 0.5f);

                int i0 = (int)x, i1 = i0 + 1;
                int j0 = (int)y, j1 = j0 + 1;
                int k0 = (int)z, k1 = k0 + 1;

                float s1 = x - i0, s0 = 1 - s1;
                float t1 = y - j0, t0 = 1 - t1;
                float u1 = z - k0, u0 = 1 - u1;

                // Direct access to array elements to improve performance
                float d0_i0j0k0 = d0[IX(i0, j0, k0)];
                float d0_i0j0k1 = d0[IX(i0, j0, k1)];
                float d0_i0j1k0 = d0[IX(i0, j1, k0)];
                float d0_i0j1k1 = d0[IX(i0, j1, k1)];
                float d0_i1j0k0 = d0[IX(i1, j0, k0)];
                float d0_i1j0k1 = d0[IX(i1, j0, k1)];
                float d0_i1j1k0 = d0[IX(i1, j1, k0)];
                float d0_i1j1k1 = d0[IX(i1, j1, k1)];

                // 3D interpolation
                d[idx] = s0 * (t0 * (u0 * d0_i0j0k0 + u1 * d0_i0j0k1) +
                               t1 * (u0 * d0_i0j1k0 + u1 * d0_i0j1k1)) +
                         s1 * (t0 * (u0 * d0_i1j0k0 + u1 * d0_i1j0k1) +
                               t1 * (u0 * d0_i1j1k0 + u1 * d0_i1j1k1));
            }
        }
    }

    set_bnd(M, N, O, b, d);

    //cudaMalloc((void **)&d_d, size);
    //cudaMemcpy(d_d, d, size, cudaMemcpyHostToDevice);   
    //launch_set_bnd_kernel(M, N, O, b, d_d);
    //cudaMemcpy(d, d_d, size, cudaMemcpyDeviceToHost);
    //cudaFree(d_d);
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
    launch_set_bnd_kernel(M, N, O, b, d_d);
    
    cudaMemcpy(d, d_d, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(d0_d, d0, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(u, u_d, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(v, v_d, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(w, w_d, size, cudaMemcpyDeviceToHost);  

    cudaFree(d_d);
    cudaFree(d0_d);
    cudaFree(u_d);
    cudaFree(v_d);
    cudaFree(w_d);    
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
    int val = M + 2;
    int val2 = N + 2;
    int y = M + 2;
    int z = (M + 2) * (N + 2); 
    int max = MAX(M, MAX(N, O));
    float invMax = 1.0f / max;
    float *u_d, *v_d, *w_d, *p_d, *div_d;

#if 0

    float *u_d, *v_d, *w_d, *p_d, *div_d;
    int size = (M + 2) * (N + 2) * (O + 2) * sizeof(float);

    cudaMalloc((void **)&u_d, size);
    cudaMalloc((void **)&v_d, size);
    cudaMalloc((void **)&w_d, size);
    cudaMalloc((void **)&p_d, size);
    cudaMalloc((void **)&div_d, size);

    cudaMemcpy(u_d, u, size, cudaMemcpyHostToDevice);
    cudaMemcpy(v_d, v, size, cudaMemcpyHostToDevice);
    cudaMemcpy(w_d, w, size, cudaMemcpyHostToDevice);
    cudaMemcpy(p_d, p, size, cudaMemcpyHostToDevice);
    cudaMemcpy(div_d, divv, size, cudaMemcpyHostToDevice);   

    dim3 blockDim(16, 16, 4);
    dim3 gridDim((M + blockDim.x - 1) / blockDim.x,
                   (N + blockDim.y - 1) / blockDim.y,
                   (O + blockDim.z - 1) / blockDim.z);
    loop1_project_kernel<<<gridDim, blockDim>>>(M, N, O, u_d, v_d, w_d, p_d, div_d);
    cudaDeviceSynchronize();

    launch_set_bnd_kernel(M, N, O, 0, div_d);
    launch_set_bnd_kernel(M, N, O, 0, p_d);
    lin_solve_kernel(M, N, O, 0, p_d, div_d, 1, 6);

    loop2_project_kernel<<<gridDim, blockDim>>>(M, N, O, u_d, v_d, w_d, p_d);
    cudaDeviceSynchronize();

    launch_set_bnd_kernel(M, N, O, 1, u_d),
    launch_set_bnd_kernel(M, N, O, 2, v_d);
    launch_set_bnd_kernel(M, N, O, 3, w_d);

    cudaMemcpy(u, u_d, size, cudaMemcpyDeviceToHost);  
    cudaMemcpy(v, v_d, size, cudaMemcpyDeviceToHost);  
    cudaMemcpy(w, w_d, size, cudaMemcpyDeviceToHost);  
    cudaMemcpy(p, p_d, size, cudaMemcpyDeviceToHost);  
    cudaMemcpy(divv, div_d, size, cudaMemcpyDeviceToHost);  

    cudaFree(u_d);
    cudaFree(v_d);
    cudaFree(w_d);
    cudaFree(p_d);
    cudaFree(div_d); 

#else

    for (int k = 1; k <= O; k++) {
      for (int j = 1; j <= N; j++) {
        for (int i = 1; i <= M; i++) {
          int idx = IX(i, j, k);

          divv[idx] = (-0.5f * (u[idx + 1] - u[idx - 1] + v[idx + y] -
                               v[idx - y] + w[idx + z] - w[idx - z])) * invMax;
          p[idx] = 0;
        }
      }
    }

    set_bnd(M, N, O, 0, divv);
    set_bnd(M, N, O, 0, p);
    //lin_solve(M, N, O, 0, p, div, 1, 6);
    
    int size = (M + 2) * (N + 2) * (O + 2) * sizeof(float);
    cudaMalloc((void **)&p_d, size);
    cudaMalloc((void **)&div_d, size);
    cudaMemcpy(p_d, p, size, cudaMemcpyHostToDevice);
    cudaMemcpy(div_d, divv, size, cudaMemcpyHostToDevice); 

    lin_solve_kernel(M, N, O, 0, p_d, div_d, 1, 6);
    cudaDeviceSynchronize();

    cudaMemcpy(p, p_d, size, cudaMemcpyDeviceToHost);  
    cudaDeviceSynchronize();

    cudaFree(p_d);
    cudaFree(div_d); 

    // Adjustment of u, v, and w without loop blocking
    for (int k = 1; k <= O; k++) {
      for (int j = 1; j <= N; j++) {
        for (int i = 1; i <= M; i++) {
          int idx = IX(i, j, k);

          u[idx] -= 0.5f * (p[idx + 1] - p[idx - 1]);
          v[idx] -= 0.5f * (p[idx + y] - p[idx - y]);
          w[idx] -= 0.5f * (p[idx + z] - p[idx - z]);
        }
      }
    }

    set_bnd(M, N, O, 1, u);
    set_bnd(M, N, O, 2, v);
    set_bnd(M, N, O, 3, w);
#endif
}

// Step function for density
void dens_step(int M, int N, int O, float *x, float *x0, float *u, float *v, float *w, float diff, float dt) {
  add_source(M, N, O, x, x0, dt);
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

  add_source(M, N, O, u, u0, dt);
  add_source(M, N, O, v, v0, dt);
  add_source(M, N, O, w, w0, dt);
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
