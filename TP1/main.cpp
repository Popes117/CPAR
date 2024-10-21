#include "EventManager.h"
#include "fluid_solver.h"
#include <iostream>
#include <vector>
#include <iostream>
#include <chrono>

#define SIZE 42

#define IX(i, j, k) ((i) + (val) * (j) + (val) * (val2) * (k))

// Globals for the grid size
static int M = SIZE;
static int N = SIZE;
static int O = SIZE;
static float dt = 0.1f;      // Time delta
static float diff = 0.0001f; // Diffusion constant
static float visc = 0.0001f; // Viscosity constant
static int val = M + 2;
static int val2 = N + 2;
static int iX = M *0.5;
static int jX = N *0.5;
static int kX = O *0.5;
static int index = IX(iX, jX, kX);
// Fluid simulation arrays
static float *u, *v, *w, *u_prev, *v_prev, *w_prev;
static float *dens, *dens_prev;

// Function to allocate simulation data
int allocate_data() {
  int size = (M + 2) * (N + 2) * (O + 2);
  u = new float[size];
  v = new float[size];
  w = new float[size];
  u_prev = new float[size];
  v_prev = new float[size];
  w_prev = new float[size];
  dens = new float[size];
  dens_prev = new float[size];

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
}

// Apply events (source or force) for the current timestep
void apply_events(const std::vector<Event> &events) {

  for (const auto &event : events) {
    if (event.type == ADD_SOURCE) {
      // Apply density source at the center of the grid
      dens[index] = event.density;
    } else if (event.type == APPLY_FORCE) {
      // Apply forces based on the event's vector (fx, fy, fz)
      u[index] = event.force.x;
      v[index] = event.force.y;
      w[index] = event.force.z;
    }
  }
}

// Function to sum the total density
float sum_density() {
  float total_density = 0.0f;
  int size = (M + 2) * (N + 2) * (O + 2);
  for (int i = 0; i < size; i++) {
    total_density += dens[i];
  }
  return total_density;
}

// Simulation loop
void simulate(EventManager &eventManager, int timesteps) {
  for (int t = 0; t < timesteps; t++) {
    // Get the events for the current timestep
    std::vector<Event> events = eventManager.get_events_at_timestamp(t);

    // Apply events to the simulation
    apply_events(events);

    // Perform the simulation steps
    vel_step(M, N, O, u, v, w, u_prev, v_prev, w_prev, visc, dt);
    dens_step(M, N, O, dens, dens_prev, u, v, w, diff, dt);
  }
}

int main() {

  //auto start = std::chrono::high_resolution_clock::now();

  // Initialize EventManager
  EventManager eventManager;
  eventManager.read_events("events.txt");

  //int val = M + 2;
  //int val2 = N + 2;

  //for(int i = 1; i < 200; i++){
  //  std::cout << "IX(1,1," << i << ") : " << IX(1,1,i) << std::endl;
  //  std::cout << "IX(1,2," << i << ") : " << IX(1,2,i) << std::endl;
  //}

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

  //auto end = std::chrono::high_resolution_clock::now();
  
  // Calcula a duração em milissegundos
  //std::chrono::duration<float, std::milli> duration = end - start;
  //std::chrono::duration<float> duration2 = end - start;
  //std::cout << "Tempo de execução: " << duration.count() << " ms" << std::endl;
  //std::cout << "Tempo de execução: " << duration2.count() << " s" << std::endl;

  return 0;
}