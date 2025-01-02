#include "EventManager.h"
#include "fluid_solver.h"
#include <iostream>
#include <vector>
#include <cuda.h>

#define SIZE 168

#define IX(i, j, k) ((i) + (M + 2) * (j) + (M + 2) * (N + 2) * (k))

#define NUM_BLOCKS 512
#define NUM_THREADS_PER_BLOCK 256
#define TOTALSIZE NUM_BLOCKS*NUM_THREADS_PER_BLOCK

// Globals for the grid size
static int M = SIZE;
static int N = SIZE;
static int O = SIZE;
static float dt = 0.1f;      // Time delta
static float diff = 0.0001f; // Diffusion constant
static float visc = 0.0001f; // Viscosity constant

// Fluid simulation arrays
static float *u, *v, *w, *u_prev, *v_prev, *w_prev;
static float *dens, *dens_prev;
float *du, *dv, *dw;
float *du_prev, *dv_prev, *dw_prev;
float *ddens, *ddens_prev;
float *changes_d, *d_max_c, *d_intermediate;

// Function to allocate simulation data
int allocate_data() {
  int size = (M + 2) * (N + 2) * (O + 2);
  int bytes = size * sizeof(float);
  const unsigned int blockSize = 512; // Pode ajustar para diferentes GPUs
  const unsigned int gridSize = (size + blockSize * 2 - 1) / (blockSize * 2);
  u = new float[size];
  v = new float[size];
  w = new float[size];
  u_prev = new float[size];
  v_prev = new float[size];
  w_prev = new float[size];
  dens = new float[size];
  dens_prev = new float[size];
  cudaMalloc((void **)&du, bytes);
  cudaMalloc((void **)&dv, bytes);
  cudaMalloc((void **)&dw, bytes);
  cudaMalloc((void **)&du_prev, bytes);
  cudaMalloc((void **)&dv_prev, bytes);
  cudaMalloc((void **)&dw_prev, bytes);
  cudaMalloc((void **)&ddens, bytes);
  cudaMalloc((void **)&ddens_prev, bytes);
  cudaMalloc((void **)&changes_d, bytes);
  cudaMalloc((void **)&d_max_c, sizeof(float));
  cudaMalloc((void **)&d_intermediate, gridSize * sizeof(float));


  if (!u || !v || !w || !u_prev || !v_prev || !w_prev || !dens || !dens_prev) {
    std::cerr << "Cannot allocate memory" << std::endl;
    return 0;
  }
  return 1;
}

// Function to clear the data (set all to zero)
void clear_data() {
  int size = (M + 2) * (N + 2) * (O + 2);
  for (int i = 0; i < size; i++) {
    u[i] = v[i] = w[i] = u_prev[i] = v_prev[i] = w_prev[i] = dens[i] =
        dens_prev[i] = 0.0f;
  }
  cudaMemcpy(du, u, (M + 2) * (N + 2) * (O + 2) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dv, v, (M + 2) * (N + 2) * (O + 2) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dw, w, (M + 2) * (N + 2) * (O + 2) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(du_prev, u_prev, (M + 2) * (N + 2) * (O + 2) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dv_prev, v_prev, (M + 2) * (N + 2) * (O + 2) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dw_prev, w_prev, (M + 2) * (N + 2) * (O + 2) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(ddens, dens, (M + 2) * (N + 2) * (O + 2) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(ddens_prev, dens_prev, (M + 2) * (N + 2) * (O + 2) * sizeof(float), cudaMemcpyHostToDevice);
}

// Free allocated memory
void free_data() {
  delete[] u;
  delete[] v;
  delete[] w;
  delete[] u_prev;
  delete[] v_prev;
  delete[] w_prev;
  delete[] dens;
  delete[] dens_prev;
  cudaFree(du);
  cudaFree(dv);
  cudaFree(dw);
  cudaFree(du_prev);
  cudaFree(dv_prev);
  cudaFree(dw_prev);
  cudaFree(ddens);
  cudaFree(ddens_prev);
  cudaFree(changes_d);
  cudaFree(d_max_c);
  cudaFree(d_intermediate);
}


__global__ void apply_events_kernel(const Event *events, int num_events, float *u, float *v, float *w, float *dens, int idx) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int e = 0; e < num_events; e++) {
            Event event = events[e];
            if (event.type == ADD_SOURCE) {
                dens[idx] = event.density;
            } else if (event.type == APPLY_FORCE) {
                u[idx] = event.force.x;
                v[idx] = event.force.y;
                w[idx] = event.force.z;
            }
        }
    }
}

// Função principal para gerenciar transferência e execução
void apply_events(const std::vector<Event> &events, int idx, float *dens, float *u, float *v, float *w) {
  int size = events.size();
  if (size == 0) return;
  Event *d_events;
  cudaMalloc((void **)&d_events, size * sizeof(Event));
  cudaMemcpy(d_events, events.data(), size * sizeof(Event), cudaMemcpyHostToDevice);

  // Lançar o kernel
  apply_events_kernel<<<1, 1>>>(d_events, size, u, v, w, dens, idx);

  cudaFree(d_events);
}

#if 0

template <unsigned int blockSize>
__global__ void reduce_sum_density(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];  // Memória compartilhada
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;
    unsigned int gridSize = blockSize * 2 * gridDim.x;

    // Inicializa memória compartilhada
    sdata[tid] = 0;

    // Soma os elementos atribuídos ao thread
    while (i < n) {
        sdata[tid] += g_idata[i];
        if (i + blockSize < n) {
            sdata[tid] += g_idata[i + blockSize];
        }
        i += gridSize;  // Incrementa para processar elementos restantes
    }
    __syncthreads();

    // Redução em memória compartilhada
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }

    if (tid < 32) {
        volatile float *vshared = sdata; // Evita leitura de memória global
        vshared[tid] += vshared[tid + 32];
        vshared[tid] += vshared[tid + 16];
        vshared[tid] += vshared[tid + 8];
        vshared[tid] += vshared[tid + 4];
        vshared[tid] += vshared[tid + 2];
        vshared[tid] += vshared[tid + 1];
    }

    // Escreve o resultado parcial na memória global
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

float sum_density(float *ddens, int size) {
    const int threadsPerBlock = 512;  // Ajustável dependendo do hardware
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    // Alocar memória para resultados intermediários
    float *d_intermediate;
    cudaMalloc(&d_intermediate, blocksPerGrid * sizeof(float));

    // Soma total
    float total_density = 0.0f;

    // Primeira chamada ao kernel
    reduce_sum_density<512><<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(ddens, d_intermediate, size);

    // Reduzir iterativamente até restar um único bloco
    while (blocksPerGrid > 1) {
        int newBlocksPerGrid = (blocksPerGrid + threadsPerBlock - 1) / threadsPerBlock;
        reduce_sum_density<512><<<newBlocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_intermediate, d_intermediate, blocksPerGrid);
        blocksPerGrid = newBlocksPerGrid;
    }

    // Copiar o resultado final para o host
    cudaMemcpy(&total_density, d_intermediate, sizeof(float), cudaMemcpyDeviceToHost);

    // Liberar memória
    cudaFree(d_intermediate);

    return total_density;
}

#else

//Function to sum the total density
float sum_density() {
  cudaMemcpy(dens, ddens, (M + 2) * (N + 2) * (O + 2) * sizeof(float), cudaMemcpyDeviceToHost);
  float total_density = 0.0f;
  int size = (M + 2) * (N + 2) * (O + 2);
  for (int i = 0; i < size; i++) {
    total_density += dens[i];
  }
  return total_density;
}
#endif 
// Simulation loop
void simulate(EventManager &eventManager, int timesteps) {
  int i = M / 2, j = N / 2, k = O / 2;
  int idx = IX(i, j, k);

  for (int t = 0; t < timesteps; t++) {
    // Get the events for the current timestep
    std::vector<Event> events = eventManager.get_events_at_timestamp(t);

    // Apply events to the simulation
    apply_events(events,idx, ddens, du, dv, dw);

    // Perform the simulation steps
    vel_step(M, N, O, du, dv, dw, du_prev, dv_prev, dw_prev, visc, dt, changes_d, d_max_c, d_intermediate);
    dens_step(M, N, O, ddens, ddens_prev, du, dv, dw, diff, dt, changes_d, d_max_c, d_intermediate);
    std::cout << "Timestep " << t << std::endl;
  }
}

int main() {
  // Initialize EventManager
  EventManager eventManager;
  eventManager.read_events("events.txt");

  // Get the total number of timesteps from the event file
  int timesteps = eventManager.get_total_timesteps();

  // Allocate and clear data
  if (!allocate_data())
    return -1;
  clear_data();

  // Run simulation with events
  simulate(eventManager, timesteps);

  // Print total density at the end of simulation
  //float total_density = sum_density(ddens, (M + 2) * (N + 2) * (O + 2));
  float total_density = sum_density();
  std::cout << "Total density after " << timesteps
            << " timesteps: " << total_density << std::endl;

  // Free memory
  free_data();

  return 0;
}