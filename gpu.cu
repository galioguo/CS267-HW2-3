#include "common.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>

#define NUM_THREADS 256

// Put any static global variables here that you will use throughout the simulation.
int blks;
int grid_size, num_bins;
double cell_size;
int *d_bin_count, *d_bin_ids, *d_part_ids;

__device__ void apply_force_gpu(particle_t& particle, particle_t& neighbor) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if (r2 > cutoff * cutoff)
        return;
    // r2 = fmax( r2, min_r*min_r );
    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
    double r = sqrt(r2);

    //
    //  very simple short-range repulsive force
    //
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

__global__ void compute_forces_gpu(particle_t* particles, int num_parts) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    particles[tid].ax = particles[tid].ay = 0;
    for (int j = 0; j < num_parts; j++)
        apply_force_gpu(particles[tid], particles[j]);
}

__global__ void move_gpu(particle_t* particles, int num_parts, double size) {

    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    particle_t* p = &particles[tid];
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x += p->vx * dt;
    p->y += p->vy * dt;

    //
    //  bounce from walls
    //
    while (p->x < 0 || p->x > size) {
        p->x = p->x < 0 ? -(p->x) : 2 * size - p->x;
        p->vx = -(p->vx);
    }
    while (p->y < 0 || p->y > size) {
        p->y = p->y < 0 ? -(p->y) : 2 * size - p->y;
        p->vy = -(p->vy);
    }
}

void init_simulation(particle_t* parts, int num_parts, double size) {
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // parts live in GPU memory
    // Do not do any particle simulation here

    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;

    grid_size = static_cast<int>(std::floor(size / cutoff)); // number of bins on one side
    num_bins = grid_size * grid_size; // number of bins
    cell_size = size / grid_size; // size of a bin

    // allocate memory on GPU
    cudaMalloc((void**)&d_bin_count, num_bins * sizeof(int));
    cudaMalloc((void**)&d_bin_ids, num_bins * sizeof(int));
    cudaMalloc((void**)&d_part_ids, num_parts * sizeof(int));
}

// CUDA, zero accelerations
__global__ void zero_accelerations(particle_t* parts, int num_parts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_parts) {
        parts[idx].ax = 0;
        parts[idx].ay = 0;
    }
}

// CUDA, compute bin counts
__global__ void compute_bin_counts(particle_t* parts, int* bin_count, int num_parts, float cell_size, int grid_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_parts) {
        int cell_col = static_cast<int>(parts[idx].x / cell_size);
        int cell_row = static_cast<int>(parts[idx].y / cell_size);
        atomicAdd(&bin_count[cell_row * grid_size + cell_col], 1);  // Atomic increment
    }
}

// CUDA, bin particles
__global__ void bin_particles(particle_t* parts, int* bin_count, int* bin_ids, int* part_ids, int num_parts, float cell_size, int grid_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_parts) {
        int cell_col = static_cast<int>(parts[idx].x / cell_size);
        int cell_row = static_cast<int>(parts[idx].y / cell_size);
        int bin_idx = cell_row * grid_size + cell_col;

        int pos = atomicAdd(&bin_count[bin_idx], 1);  // Get position in bin and increment
        part_ids[bin_ids[bin_idx] + pos] = idx;  // Store particle ID at correct position
    }
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    
    // Reset bin counts
    cudaMemset(d_bin_count, 0, num_bins * sizeof(int));
    cudaMemset(d_bin_ids, -1, num_bins * sizeof(int));
    cudaMemset(d_part_ids, 0, num_parts * sizeof(int));

    zero_accelerations<<<blks, NUM_THREADS>>>(parts, num_parts);
    
    // bin particles
    // acquire bin counts
    compute_bin_counts<<<blks, NUM_THREADS>>>(parts, d_bin_count, num_parts, cell_size, grid_size);

    // compute prefix sums
    thrust::device_ptr<int> dev_bin_count(d_bin_count);
    thrust::device_ptr<int> dev_bin_ids(d_bin_ids);
    thrust::exclusive_scan(dev_bin_count, dev_bin_ids + num_bins, dev_bin_ids);  // Exclusive scan

    // write over all elements in bin_count to 0 
    // cudaMemset(d_bin_count, 0, num_bins * sizeof(int));
    
    // binning
    // bin_particles<<<blks, NUM_THREADS>>>(parts, d_bin_count, d_bin_ids, d_part_ids, num_parts, cell_size, grid_size);

    // copy back to CPU for debug
    int *bin_count = new int[num_bins];
    int *bin_ids = new int[num_bins];
    int *part_ids = new int[num_parts];
    cudaMemcpy(bin_count, d_bin_count, num_bins * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(bin_ids, d_bin_count, num_bins * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(part_ids, d_bin_count, num_parts * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Bin Counts: " << "\n";
    for (int bin_row = grid_size - 1; bin_row >= 0; --bin_row) {
        for (int bin_col = 0; bin_col < grid_size; ++bin_col) {
            std::cout << bin_count[bin_row * grid_size + bin_col] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "Bin IDs: " << "\n";
    for (int bin_row = grid_size - 1; bin_row >= 0; --bin_row) {
        for (int bin_col = 0; bin_col < grid_size; ++bin_col) {
            std::cout << bin_ids[bin_row * grid_size + bin_col] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "Particle IDs: " << "\n";
    for (int part_i = 0; part_i < num_parts; ++part_i) {
        std::cout << part_ids[part_i] << " ";
        std::cout << "\n";
    }

    // COMPUTE FORCES
    
    // interiors

    // left edge 

    // right edge

    // top edge

    // bottom edge

    // left-top corner

    // right-top corner

    // left-bottom corner

    // right-bottom corner 

    // MOVE PARTICLES
    
    // Compute forces
    //compute_forces_gpu<<<blks, NUM_THREADS>>>(parts, num_parts);

    // Move particles
    //move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);
}
