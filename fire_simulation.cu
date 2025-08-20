#include "simulation.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <curand_kernel.h>
#include <cuda_gl_interop.h>

// ====================================================================================
// Strutture Dati Interne al Modulo di Simulazione
// ====================================================================================
typedef struct {
    float posX, posY, posZ;
    float velX, velY, velZ;
    float lifetime;
} ParticleState;

// Questa è la nostra "struct opaca". Qui la definiamo per davvero.
// Contiene tutti i puntatori e i dati necessari alla simulazione.
struct Simulation_t {
    int numParticles;
    int gridSize;
    int blockSize;

    ParticleState* d_particles;
    curandState* d_randStates;
    struct cudaGraphicsResource* vbo_resource; // Puntatore alla risorsa VBO
};

// ====================================================================================
// Kernels (Invariati)
// ====================================================================================
__global__ void initCurandKernel(curandState* states, int numParticles, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;
    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void updateParticlesKernel(ParticleState* states, ParticleVertex* vbo_ptr, int numParticles, float dt, curandState* randStates) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    curandState localState = randStates[idx];
    ParticleState p = states[idx];

    if (p.lifetime > 0.0f) {
        p.velY -= 2.0f * dt;
        p.posX += p.velX * dt;
        p.posY += p.velY * dt;
        p.lifetime -= dt;
    }
    else {
        p.velX = (curand_uniform(&localState) - 0.5f) * 0.2f;
        p.velY = curand_uniform(&localState) * 1.5f + 1.0f;
        p.posX = 0.0f;
        p.posY = -0.8f;
        p.lifetime = 1.0f + curand_uniform(&localState) * 2.0f;
        p.posX += p.velX * dt;
        p.posY += p.velY * dt;
    }

    states[idx] = p;
    vbo_ptr[idx].x = p.posX;
    vbo_ptr[idx].y = p.posY;
    vbo_ptr[idx].z = 0.0f;
    randStates[idx] = localState;
}

// ====================================================================================
// Implementazione delle Funzioni Pubbliche (definite in simulation.h)
// ====================================================================================

Simulation* simulation_create(int numParticles) {
    Simulation* sim = (Simulation*)malloc(sizeof(Simulation));
    if (sim == NULL) return NULL;

    sim->numParticles = numParticles;
    sim->blockSize = 256;
    sim->gridSize = (numParticles + sim->blockSize - 1) / sim->blockSize;
    sim->vbo_resource = NULL;

    printf("Simulazione creata per %d particelle.\n", numParticles);
    printf("Configurazione Kernel: %d blocchi, %d thread per blocco.\n", sim->gridSize, sim->blockSize);

    // Allocazione memoria per lo stato delle particelle
    size_t states_size = numParticles * sizeof(ParticleState);
    ParticleState* h_states = (ParticleState*)malloc(states_size);
    for (int i = 0; i < numParticles; i++) h_states[i].lifetime = -1.0f;

    cudaMalloc((void**)&sim->d_particles, states_size);
    cudaMemcpy(sim->d_particles, h_states, states_size, cudaMemcpyHostToDevice);
    free(h_states);

    // Allocazione e inizializzazione di cuRAND
    cudaMalloc((void**)&sim->d_randStates, numParticles * sizeof(curandState));
    initCurandKernel << <sim->gridSize, sim->blockSize >> > (sim->d_randStates, numParticles, time(NULL));

    return sim;
}

void simulation_destroy(Simulation* sim) {
    if (sim == NULL) return;

    // Unregister VBO se era stato registrato
    if (sim->vbo_resource) {
        cudaGraphicsUnregisterResource(sim->vbo_resource);
    }

    cudaFree(sim->d_particles);
    cudaFree(sim->d_randStates);
    free(sim);
    printf("Simulazione distrutta e memoria liberata.\n");
}

void simulation_update(Simulation* sim, float dt) {
    if (sim == NULL || sim->vbo_resource == NULL) return;

    // Mappa il VBO per ottenere un puntatore CUDA
    ParticleVertex* d_vbo_ptr;
    cudaGraphicsMapResources(1, &sim->vbo_resource, 0);
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void**)&d_vbo_ptr, &num_bytes, sim->vbo_resource);

    // Lancia il kernel
    updateParticlesKernel << <sim->gridSize, sim->blockSize >> > (sim->d_particles, d_vbo_ptr, sim->numParticles, dt, sim->d_randStates);

    // Unmap del VBO
    cudaGraphicsUnmapResources(1, &sim->vbo_resource, 0);
}

void simulation_register_vbo(Simulation* sim, unsigned int vbo_id) {
    if (sim == NULL) return;
    // Registra il VBO di OpenGL per l'interoperabilità
    cudaGraphicsGLRegisterBuffer(&sim->vbo_resource, vbo_id, cudaGraphicsRegisterFlagsNone);
}

struct cudaGraphicsResource* simulation_get_vbo_resource(Simulation* sim) {
    if (sim == NULL) return NULL;
    return sim->vbo_resource;
}