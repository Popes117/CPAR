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

// Function to allocate simulation data
int allocate_data() {
  int size = (M + 2) * (N + 2) * (O + 2);
  int bytes = size * sizeof(float);
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
}


__global__ void apply_events_kernel(Event *events, int num_events, int center_idx, float *du, float *dv, float *dw, float *ddens) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Cada thread trata de um evento
    if (idx < num_events) {
        Event event = events[idx];

        if (event.type == ADD_SOURCE) {
            // Aplicar densidade no centro
            ddens[center_idx] = event.density;
        } else if (event.type == APPLY_FORCE) {
            // Aplicar forças no centro
            du[center_idx] = event.force.x;
            dv[center_idx] = event.force.y;
            dw[center_idx] = event.force.z;
        }
    }
}


// Apply events (source or force) for the current timestep
void apply_events(const std::vector<Event> &events,int idx, float *dens, float *u, float *v, float *w) {

  int size = events.size();
  Event *d_events;
  cudaMalloc((void **)&d_events, size * sizeof(Event));
  cudaMemcpy(d_events, events.data(), size * sizeof(Event), cudaMemcpyHostToDevice);
  int threads_per_block = 256;
  int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;

  // Lançar o kernel
  apply_events_kernel<<<blocks_per_grid, threads_per_block>>>(d_events, size, idx, u, v, w, dens);

  cudaFree(d_events);

}

// Function to sum the total density
float sum_density() {
  cudaMemcpy(dens, ddens, (M + 2) * (N + 2) * (O + 2) * sizeof(float), cudaMemcpyDeviceToHost);
  float total_density = 0.0f;
  int size = (M + 2) * (N + 2) * (O + 2);
  for (int i = 0; i < size; i++) {
    total_density += dens[i];
    //if (dens[i] > 0.1f) {
    //  printf("dens[%i]: %f\n", i, dens[i]);
    //}
  }
  return total_density;
}

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
    vel_step(M, N, O, du, dv, dw, du_prev, dv_prev, dw_prev, visc, dt);
    dens_step(M, N, O, ddens, ddens_prev, du, dv, dw, diff, dt);
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
  float total_density = sum_density();
  std::cout << "Total density after " << timesteps
            << " timesteps: " << total_density << std::endl;

  // Free memory
  free_data();

  return 0;
}