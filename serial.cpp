#include "common.h"
#include <cmath>
#include <cstring>
#include <iostream>

int grid_size = 0; // number of bins on one side
int num_bins = 0; // number of bins
double cell_size = 0.0; // size of a bin

static int* bin_count = nullptr;
static int* bin_ids = nullptr;
static int* part_ids = nullptr;

// Apply the force from neighbor to particle
void apply_force(particle_t& particle, particle_t& neighbor) {
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);

    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

// Integrate the ODE
void move(particle_t& p, double size) {
    // Slightly simplified Velocity Verlet integration
    // Conserves energy better than explicit Euler method
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    // Bounce from walls
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }

    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}


void init_simulation(particle_t* parts, int num_parts, double size) {
	// You can use this space to initialize static, global data objects
    // that you may need. This function will be called once before the
    // algorithm begins. Do not do any particle simulation here

    grid_size = static_cast<int>(std::floor(size / cutoff)); // number of bins on one side
    num_bins = grid_size * grid_size; // number of bins
    cell_size = size / grid_size; // size of a bin

    bin_count = new int[num_bins]();
    bin_ids = new int[num_bins]();
    part_ids = new int[num_parts]();

}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    
    memset(bin_count, 0, num_bins * sizeof(int));
    memset(bin_ids, 0, num_bins * sizeof(int));
    memset(part_ids, 0, num_parts * sizeof(int));
    
    // overwrite accelerations to 0
    for (int i = 0; i < num_parts; i++) {
        parts[i].ax = 0;
        parts[i].ay = 0; 
    }
    
    // bin particles
    // acquire bin counts
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
        bin_ids[i] = bin_ids[i - 1] + bin_count[i - 1];
    }

    // write over all elements in bin_count to 0 
    memset(bin_count, 0, num_bins * sizeof(int));
    
    // binning
    for (int i = 0; i < num_parts; ++i) {
        // find relevant bin coordinates
        int cell_col = static_cast<int>(parts[i].x / cell_size);
        int cell_row = static_cast<int>(parts[i].y / cell_size); 
        // write particle id in the correct place
        part_ids[bin_ids[cell_row * grid_size + cell_col] + bin_count[cell_row * grid_size + cell_col]] = i;
        // iterate count in bin by 1
        bin_count[cell_row * grid_size + cell_col] += 1; 
    }
    
    // COMPUTE FORCES
    // internal blocks 
    
    for (int cell_row = 1; cell_row < grid_size - 1; ++cell_row) {
        for (int cell_col = 1; cell_col < grid_size - 1; ++cell_col) {
            int self_cell = cell_row * grid_size + cell_col;
            for (int self_i = 0; self_i < bin_count[self_cell]; ++self_i) {
                for (int di = -1; di <= 1; ++di) {
                    for (int dj = -1; dj <= 1; ++dj) {
                        int neig_cell = (cell_row + di) * grid_size + (cell_col + dj); 
                        for (int neig_i = 0; neig_i < bin_count[neig_cell]; ++neig_i) {
                            apply_force(parts[part_ids[self_i + bin_ids[self_cell]]], parts[part_ids[neig_i + bin_ids[neig_cell]]]);
                        }
                    }
                }
            }
        }
    }
    
    // left edge 
    int cell_col = 0; 
    for (int cell_row = 1; cell_row < grid_size - 1; ++cell_row) {
        int self_cell = cell_row * grid_size + cell_col;
        for (int self_i = 0; self_i < bin_count[self_cell]; ++self_i) {
            for (int di = -1; di <= 1; ++di) {
                for (int dj = 0; dj <= 1; ++dj) {
                    int neig_cell = (cell_row + di) * grid_size + (cell_col + dj); 
                    for (int neig_i = 0; neig_i < bin_count[neig_cell]; ++neig_i) {
                        apply_force(parts[part_ids[self_i + bin_ids[self_cell]]], parts[part_ids[neig_i + bin_ids[neig_cell]]]);
                    }
                }
            }
        }
    }
    
    // right edge
    cell_col = grid_size - 1; 
    for (int cell_row = 1; cell_row < grid_size - 1; ++cell_row) {
        int self_cell = cell_row * grid_size + cell_col;
        for (int self_i = 0; self_i < bin_count[self_cell]; ++self_i) {
            for (int di = -1; di <= 1; ++di) {
                for (int dj = -1; dj <= 0; ++dj) {
                    int neig_cell = (cell_row + di) * grid_size + (cell_col + dj); 
                    for (int neig_i = 0; neig_i < bin_count[neig_cell]; ++neig_i) {
                        apply_force(parts[part_ids[self_i + bin_ids[self_cell]]], parts[part_ids[neig_i + bin_ids[neig_cell]]]);
                    }
                }
            }
        }
    }
    
    // top edge
    int cell_row = grid_size - 1; 
    for (int cell_col = 1; cell_col < grid_size - 1; ++cell_col) {
        int self_cell = cell_row * grid_size + cell_col;
        for (int self_i = 0; self_i < bin_count[self_cell]; ++self_i) {
            for (int di = -1; di <= 0; ++di) {
                for (int dj = -1; dj <= 1; ++dj) {
                    int neig_cell = (cell_row + di) * grid_size + (cell_col + dj); 
                    for (int neig_i = 0; neig_i < bin_count[neig_cell]; ++neig_i) {
                        apply_force(parts[part_ids[self_i + bin_ids[self_cell]]], parts[part_ids[neig_i + bin_ids[neig_cell]]]);
                    }
                }
            }
        }
    }
    
    // bottom edge
    cell_row = 0; 
    for (int cell_col = 1; cell_col < grid_size - 1; ++cell_col) {
        int self_cell = cell_row * grid_size + cell_col;
        for (int self_i = 0; self_i < bin_count[self_cell]; ++self_i) {
            for (int di = 0; di <= 1; ++di) {
                for (int dj = -1; dj <= 1; ++dj) {
                    int neig_cell = (cell_row + di) * grid_size + (cell_col + dj); 
                    for (int neig_i = 0; neig_i < bin_count[neig_cell]; ++neig_i) {
                        apply_force(parts[part_ids[self_i + bin_ids[self_cell]]], parts[part_ids[neig_i + bin_ids[neig_cell]]]);
                    }
                }
            }
        }
    }
    
    // left-top corner
    cell_row = grid_size - 1;
    cell_col = 0;
    int self_cell = cell_row * grid_size + cell_col;
    for (int self_i = 0; self_i < bin_count[self_cell]; ++self_i) {
        for (int di = -1; di <= 0; ++di) {
            for (int dj = 0; dj <= 1; ++dj) {
                int neig_cell = (cell_row + di) * grid_size + (cell_col + dj); 
                for (int neig_i = 0; neig_i < bin_count[neig_cell]; ++neig_i) {
                    apply_force(parts[part_ids[self_i + bin_ids[self_cell]]], parts[part_ids[neig_i + bin_ids[neig_cell]]]);
                }
            }
        }
    }

    // right-top corner
    cell_row = grid_size - 1;
    cell_col = grid_size - 1;
    self_cell = cell_row * grid_size + cell_col;
    for (int self_i = 0; self_i < bin_count[self_cell]; ++self_i) {
        for (int di = -1; di <= 0; ++di) {
            for (int dj = -1; dj <= 0; ++dj) {
                int neig_cell = (cell_row + di) * grid_size + (cell_col + dj); 
                for (int neig_i = 0; neig_i < bin_count[neig_cell]; ++neig_i) {
                    apply_force(parts[part_ids[self_i + bin_ids[self_cell]]], parts[part_ids[neig_i + bin_ids[neig_cell]]]);
                }
            }
        }
    }

    // left-bottom corner
    cell_row = 0;
    cell_col = 0;
    self_cell = cell_row * grid_size + cell_col;
    for (int self_i = 0; self_i < bin_count[self_cell]; ++self_i) {
        for (int di = 0; di <= 1; ++di) {
            for (int dj = 0; dj <= 1; ++dj) {
                int neig_cell = (cell_row + di) * grid_size + (cell_col + dj); 
                for (int neig_i = 0; neig_i < bin_count[neig_cell]; ++neig_i) {
                    apply_force(parts[part_ids[self_i + bin_ids[self_cell]]], parts[part_ids[neig_i + bin_ids[neig_cell]]]);
                }
            }
        }
    }

    // right-bottom corner 
    cell_row = 0;
    cell_col = grid_size - 1;
    self_cell = cell_row * grid_size + cell_col;
    for (int self_i = 0; self_i < bin_count[self_cell]; ++self_i) {
        for (int di = 0; di <= 1; ++di) {
            for (int dj = -1; dj <= 0; ++dj) {
                int neig_cell = (cell_row + di) * grid_size + (cell_col + dj); 
                for (int neig_i = 0; neig_i < bin_count[neig_cell]; ++neig_i) {
                    apply_force(parts[part_ids[self_i + bin_ids[self_cell]]], parts[part_ids[neig_i + bin_ids[neig_cell]]]);
                }
            }
        }
    }
    
    // Move Particles
    for (int i = 0; i < num_parts; ++i) {
        move(parts[i], size);
    }
}
