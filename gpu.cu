#include "common.h"
#include <cuda.h>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <iostream>
#include <cstring>

#define NUM_THREADS 256

// Put any static global variables here that you will use throughout the simulation.
int blks;

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
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // parts live in GPU memory
    // Rewrite this function

    int grid_size = static_cast<int>(std::floor(size / cutoff)); // number of bins on one side
    int num_bins = grid_size * grid_size; // number of bins
    double cell_size = size / grid_size; // size of a bin
    
    // BIN PARTICLES
    // acquire bin counts
    int* bin_count = new int[num_bins];
    int* bin_ids = new int[num_bins];
    int* part_ids = new int[num_parts];
    
    for (int i = 0; i < num_parts; ++i) {
        // find relevant bin coordinates
        int cell_col = static_cast<int>(parts[i].x / cell_size);
        int cell_row = static_cast<int>(parts[i].y / cell_size); 
        // iterate count in bin by 1
        bin_count[cell_row * grid_size + cell_col] += 1; 
    }

    // compute prefix sums 
    bin_ids[0] = 0;  // First element is always 0
    for (int i = 1; i < num_bins; i++) {
        bin_ids[i] = bin_count[i - 1] + bin_count[i - 1];
    }

    /*
    thrust::device_vector<int> d_bin_count(bin_count, bin_count + num_bins);
    thrust::exclusive_scan(d_bin_count.begin(), d_bin_count.end(), d_bin_count.begin());
    thrust::host_vector<int> bin_ids = d_bin_count;
    */

    // write over all elements in bin_count to 0 
    memset(bin_count, 0, sizeof(bin_count));
    
    // bin particles
    for (int i = 0; i < num_parts; ++i) {
        // find relevant bin coordinates
        int cell_col = static_cast<int>(parts[i].x / cell_size);
        int cell_row = static_cast<int>(parts[i].y / cell_size); 
        // write particle id in the correct place
        part_ids[bin_ids[cell_row * grid_size + cell_col] + bin_count[cell_row * grid_size + cell_col]] = i;
        // iterate count in bin by 1
        bin_count[cell_row * grid_size + cell_col] += 1; 
    }
    
    // Compute forces
    //compute_forces_gpu<<<blks, NUM_THREADS>>>(parts, num_parts);

    // Move particles
    //move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);
}
