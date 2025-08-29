#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <curand_kernel.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "fast_math.cuh"

#define BLOCK_SIZE 1024
#define MAX_NEIGHBORS 32      // Numero massimo di vicini in shared memory
#define CELL_SIZE 1.0f        // Dimensione di una cella della griglia
#define TILE_WIDTH 32
#define TILE_HEIGHT 16

//#define SEQUENTIAL_BASELINE
#define GPU_NAIVE_PARALLEL
//#define OPTIMIZED_DIVERGENCE_VERSION
//#define OPTIMIZED_DIVERGENCE_2

// Stato particella
struct Particle {
    float posX, posY, posZ;
    float velX, velY, velZ;
    float lifetime;
    unsigned int rand_state;
};

// Dati per rendering particella
typedef struct {
    float x, y, z;
    float r, g, b, a;
    float lifetime;
} ParticleVertex;

float get_random_float() {
    return (float)rand() / (float)RAND_MAX;
}

void initializeParticles(Particle* particles, int numParticles) {
    for (int i = 0; i < numParticles; i++) {
        float random_val = (float)rand() / (float)RAND_MAX;
        particles[i].lifetime = random_val * 3.0f;
        particles[i].posX = 0.0f;
        particles[i].posY = 0.0f;
        particles[i].posZ = 0.0f;
        particles[i].velX = 0.0f;
        particles[i].velY = 0.0f;
        particles[i].velZ = 0.0f;
        particles[i].rand_state = rand() + 1;
#ifdef OPTIMIZED_DIVERGENCE_2
        particles[i].is_dead = 0;
#endif
    }
}

// Kernel per init generatore numeri random
// Comportamento particelle INDIPENDENTE da altre
__global__ void initCurandKernel(curandState* states, int numParticles, unsigned long long seed) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) {
        return;
    }
    curand_init(seed, idx, 0, &states[idx]);
}

__device__ int3 positionToCell(float3 pos) {
    return make_int3(floor(pos.x / CELL_SIZE),
        floor(pos.y / CELL_SIZE),
        floor(pos.z / CELL_SIZE));
}

//Aggiornamento stato particelle
#ifdef SEQUENTIAL_BASELINE
void updateParticlesCPU(Particle* particles, int numParticles, float dt, float time) 
{

    for (int idx = 0; idx < numParticles; idx++) {
        Particle p = particles[idx];

        //Particella viva
        if (p.lifetime > 0.0f) {

            // Forza di convergenza, per mantenere forma della fiamma
            p.velX -= p.posX * 1.2f * dt;

            // Turbolenza
            float turbulence = sin(p.posY * 3.0f + time * 1.5f + p.posX * 2.0f)
                + cos(p.posY * 5.0f + time * 2.5f);
            p.velX += turbulence * 0.4f * dt;

            // Swirl
            float swirl = 0.3f * sin(time * 2.0f + p.posY * 4.0f);
            p.velX += swirl * dt;

            // Spinta verso l’alto, aria calda che sale
            p.velY += 0.8f * dt;

            // Resistenza dell'aria
            p.velX *= 0.985f;
            p.velY *= 0.992f;

            // Aggiornamento posizione
            p.posX += p.velX * dt;
            p.posY += p.velY * dt;
            // Diminuzione tempo di vita di delta t
            p.lifetime -= dt;

        }
        // Rinascita
        else {
            // Posizione iniziale rinascita
            p.posX = (get_random_float() - 0.5f) * 0.8f;
            p.posY = -0.8f + get_random_float() * 0.05f;
            p.posZ = 0.0f;


            // Velocità iniziale rinascita
            p.velX = (get_random_float() - 0.5f) * 0.8f;
            p.velY = get_random_float() * 1.2f + 1.0f;
            p.velZ = 0.0f;

            // Tempo di vita della particella
            p.lifetime = 0.8f + get_random_float() * 1.2f;
        }

        // Particella savlata in array
        particles[idx] = p;
    }
}
#endif

#ifdef  GPU_NAIVE_PARALLEL
__global__ void updateParticlesKernel(Particle* particles, ParticleVertex* vbo_ptr,
    int numParticles, float dt, curandState* randStates, float time, int gridWidth)
{
    // Calcolo indice globale thread
    int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    int global_y = blockIdx.y * blockDim.y + threadIdx.y;
   
    int idx = global_y * gridWidth + global_x;

    curandState localState = randStates[idx];
    Particle p = particles[idx];

    //Particella viva
    if (p.lifetime > 0.0f) 
    {

        // Forza di convergenza, per mantenere forma della fiamma
        p.velX -= p.posX * 1.2f * dt;

        // Turbolenza
        float turbulence = sin(p.posY * 3.0f + time * 1.5f + p.posX * 2.0f)
            + cos(p.posY * 5.0f + time * 2.5f);
        p.velX += turbulence * 0.4f * dt;

        // Swirl
        float swirl = 0.3f * sin(time * 2.0f + p.posY * 4.0f);
        p.velX += swirl * dt;

        // Spinta verso l’alto, aria calda che sale
        p.velY += 0.8f * dt;

        // Resistenza dell'aria
        p.velX *= 0.985f;
        p.velY *= 0.992f;

        // Aggiornamento posizione
        p.posX += p.velX * dt;
        p.posY += p.velY * dt;

        // Diminuzione tempo di vita di delta t
        p.lifetime -= dt;

    }
    // Rinascita
    else 
    {
        // Posizione iniziale rinascita
        p.posX = (curand_uniform(&localState) - 0.5f) * 0.8f;
        p.posY = -0.8f + curand_uniform(&localState) * 0.05f;
        p.posZ = 0.0f;

        // Velocità iniziale rinascita
        p.velX = (curand_uniform(&localState) - 0.5f) * 0.8f;
        p.velY = curand_uniform(&localState) * 1.2f + 1.0f;
        p.velZ = 0.0f;

        // Tempo di vita della particella
        p.lifetime = 0.8f + curand_uniform(&localState) * 1.2f;
    }

    particles[idx] = p;
    randStates[idx] = localState;
}

#endif
#ifdef OPTIMIZED_DIVERGENCE_VERSION
__global__ void updateParticlesKernel(Particle* particles, ParticleVertex* vbo_ptr,
    int numParticles, float dt, curandState* randStates, float time, int gridWidth)
{
    // Shared memory per i dati delle particelle
    __shared__ Particle tile[TILE_HEIGHT][TILE_WIDTH];

    // --- NUOVO: Shared memory per gestire la divergenza ---
    __shared__ int dead_particle_indices[BLOCK_SIZE];
    __shared__ int dead_count;

    // Inizializza il contatore delle particelle morte (solo un thread deve farlo)
    if (threadIdx.x == 0 && threadIdx.y == 0) 
    {
        dead_count = 0;
    }

    // Indici
    int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    int linear_global_idx = global_y * gridWidth + global_x;

    int local_x = threadIdx.x;
    int local_y = threadIdx.y;
    int linear_local_idx = local_y * TILE_WIDTH + local_x;

    // Caricamento in shared memory
    if (linear_global_idx < numParticles) {
        tile[local_y][local_x] = particles[linear_global_idx];
    }
    __syncthreads();

    // ==========================================================
    // FASE 1: IDENTIFICAZIONE E CONTEGGIO DELLE PARTICELLE MORTE
    // ==========================================================
    if (linear_global_idx < numParticles) {
        Particle p = tile[local_y][local_x];
        if (p.lifetime <= 0.0f) {
            // Se la particella è morta, aggiungi il suo indice alla lista condivisa
            // atomicAdd restituisce il vecchio valore del contatore
            int index_in_dead_list = atomicAdd(&dead_count, 1);
            dead_particle_indices[index_in_dead_list] = linear_global_idx;
        }
    }

    // Sincronizza per assicurarsi che dead_count e la lista siano completi
    __syncthreads();

    // ==========================================================
    // FASE 2: LAVORO PARTIZIONATO SENZA DIVERGENZA
    // ==========================================================

    // I primi 'dead_count' thread gestiscono la rinascita
    if (linear_local_idx < dead_count) {
        int particle_to_revive_idx = dead_particle_indices[linear_local_idx];

        Particle p = particles[particle_to_revive_idx];
        curandState localState = randStates[particle_to_revive_idx];

        // Posizione iniziale rinascita
        p.posX = (curand_uniform(&localState) - 0.5f) * 0.8f;
        p.posY = -0.8f + curand_uniform(&localState) * 0.05f;
        p.posZ = 0.0f;

        // Velocità iniziale rinascita
        p.velX = (curand_uniform(&localState) - 0.5f) * 0.8f;
        p.velY = curand_uniform(&localState) * 1.2f + 1.0f;
        p.velZ = 0.0f;

        // Tempo di vita della particella
        p.lifetime = 0.8f + curand_uniform(&localState) * 1.2f;

        particles[particle_to_revive_idx] = p;
        randStates[particle_to_revive_idx] = localState;
    }
    // Tutti gli altri thread gestiscono l'aggiornamento
    else {
        if (linear_global_idx < numParticles) {
            Particle p = tile[local_y][local_x];

            // Lavora solo se la particella è viva
            if (p.lifetime > 0.0f) {

                // Forza di convergenza, per mantenere forma della fiamma
                p.velX -= p.posX * 1.2f * dt;

                // Turbolenza
                float turbulence = sin(p.posY * 3.0f + time * 1.5f + p.posX * 2.0f)
                    + cos(p.posY * 5.0f + time * 2.5f);
                p.velX += turbulence * 0.4f * dt;

                // Swirl
                float swirl = 0.3f * sin(time * 2.0f + p.posY * 4.0f);
                p.velX += swirl * dt;

                // Spinta verso l’alto, aria calda che sale
                p.velY += 0.8f * dt;

                // Resistenza dell'aria
                p.velX *= 0.985f;
                p.velY *= 0.992f;

                // Aggiornamento posizione
                p.posX += p.velX * dt;
                p.posY += p.velY * dt;

                // Diminuzione tempo di vita di delta t
                p.lifetime -= dt;

                particles[linear_global_idx] = p;
            }
        }
    }

}
#endif

#ifdef OPTIMIZED_DIVERGENCE_2
// Il numero di sotto-passi da eseguire per ogni chiamata al kernel.
// 2 o 3 è un buon punto di partenza. Puoi sperimentare con questo valore.
#define SUB_STEPS 2 

__global__ void fireUpdateAndRespawnKernel(
    Particle* __restrict__ particles,
    int numParticles,
    float dt,
    float time)
    //int gridWidth)
{
    /*nt gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = gy * gridWidth + gx;
    if (idx >= numParticles) return;*/
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;


    // 1. LEGGI DALLA MEMORIA GLOBALE (UNA SOLA VOLTA)
    Particle p = particles[idx];
    unsigned int local_rand_state = p.rand_state;

    const float sub_dt = dt / (float)SUB_STEPS;

    // 2. ESEGUI PIÙ CALCOLI NEI REGISTRI
    // #pragma unroll dice al compilatore di "srotolare" questo loop,
    // eliminando il costo del loop stesso e rendendolo più veloce.
#pragma unroll
    for (int i = 0; i < SUB_STEPS; ++i) {

        // Calcola il tempo corrente per questo sotto-passo per una fisica corretta
        float current_time = time + (float)i * sub_dt;

        // --- Logica di Simulazione (identica a prima, ma ora è dentro un loop) ---

        // Respawn (usa sub_dt per i calcoli se necessario, anche se qui non serve)
        if (p.lifetime <= 0.0f) {
            p.posX = (random_float(local_rand_state) - 0.5f) * 0.2f;
            p.posY = -0.8f + random_float(local_rand_state) * 0.1f;
            p.posZ = 0.0f;

            p.velX = (random_float(local_rand_state) - 0.5f) * 0.8f;
            p.velY = 1.8f + random_float(local_rand_state) * 1.2f;
            p.velZ = 0.0f;

            p.lifetime = 1.0f + random_float(local_rand_state) * 1.5f;
        }

        // Fisica (usa sub_dt e current_time)
        p.velX -= p.posX * 1.2f * sub_dt;
        p.velY += 0.8f * sub_dt;

        float turbulence = fast_sin(p.posY * 3.0f + current_time * 1.5f + p.posX * 2.0f)
            + fast_cos(p.posY * 5.0f + current_time * 2.5f);
        p.velX += turbulence * 0.4f * sub_dt;

        float swirl = 0.3f * fast_sin(current_time * 2.0f + p.posY * 4.0f);
        p.velX += swirl * sub_dt;

        p.velX *= 0.985f;
        p.velY *= 0.992f;

        p.posX += p.velX * sub_dt;
        p.posY += p.velY * sub_dt;

        p.lifetime -= sub_dt;

        if (p.posY > 20.0f) {
            p.lifetime = -1.0f;
        }
    }

    // 3. SCRIVI NELLA MEMORIA GLOBALE (UNA SOLA VOLTA)
    p.rand_state = local_rand_state; // <-- CAMBIAMENTO QUI
    particles[idx] = p;
}
__global__ void updateVBOKernel(const Particle* __restrict__ particles, ParticleVertex* __restrict__ vbo_ptr, int numParticles)//,int gridWidth)
{
    /*int gx = blockIdx.x * blockDim.x + threadIdx.x;
            int gy = blockIdx.y * blockDim.y + threadIdx.y;
            int idx = gy * gridWidth + gx;
            if (idx >= numParticles) return;*/
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;


    Particle p = particles[idx];
    vbo_ptr[idx].x = p.posX;
    vbo_ptr[idx].y = p.posY;
    vbo_ptr[idx].z = p.posZ;
    vbo_ptr[idx].lifetime = p.lifetime;
}











__global__ void updateAndRespawnKernel(
    Particle* __restrict__ particles,
    int numParticles,
    float dt,
    float time,
    int gridWidth)
{
    /*int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = gy * gridWidth + gx;
    if (idx >= numParticles) return;*/
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    // 1. Carica la particella nei registri
    Particle p = particles[idx];

    // 2. Se la particella è viva, aggiorna la sua fisica
    if (p.lifetime > 0.0f) {
        // Forza di convergenza
        p.velX -= p.posX * 1.2f * dt;
        // Turbolenza (con la LUT)
        float turbulence = fast_sin(p.posY * 3.0f + time * 1.5f + p.posX * 2.0f)
            + fast_cos(p.posY * 5.0f + time * 2.5f);
        p.velX += turbulence * 0.4f * dt;
        // Swirl (con la LUT)
        float swirl = 0.3f * fast_sin(time * 2.0f + p.posY * 4.0f);
        p.velX += swirl * dt;
        // Spinta verso l’alto
        p.velY += 0.8f * dt;
        // Resistenza
        p.velX *= 0.985f;
        p.velY *= 0.992f;
        // Posizione
        p.posX += p.velX * dt;
        p.posY += p.velY * dt;
        // Invecchiamento
        p.lifetime -= dt;
    }

    // 3. Se la particella è morta (o è appena morta), falla rinascere
    if (p.lifetime <= 0.0f) {
        // Usa il nostro PRNG veloce. Passiamo p.rand_state per riferimento
        // in modo che venga aggiornato per il prossimo utilizzo.
        p.posX = (random_float(p.rand_state) - 0.5f) * 0.8f;
        p.posY = -0.8f + random_float(p.rand_state) * 0.05f;
        p.posZ = 0.0f;

        p.velX = (random_float(p.rand_state) - 0.5f) * 0.8f;
        p.velY = random_float(p.rand_state) * 1.2f + 1.0f;
        p.velZ = 0.0f;

        p.lifetime = 0.8f + random_float(p.rand_state) * 1.2f;
    }

    // 4. Scrivi lo stato finale della particella in memoria globale
    particles[idx] = p;
}




















__global__ void updateAliveKernel(
    Particle* __restrict__ particles,
    int numParticles,
    float dt,
    float time,
    int gridWidth)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = gy * gridWidth + gx;
    if (idx >= numParticles) return;


    Particle p = particles[idx];


    if (!p.is_dead) {
        // Forza di convergenza
        p.velX -= p.posX * 1.2f * dt;
        // Turbolenza
        float turbulence = fast_sin(p.posY * 3.0f + time * 1.5f + p.posX * 2.0f)
            + fast_cos(p.posY * 5.0f + time * 2.5f);
        p.velX += turbulence * 0.4f * dt;
        // Swirl
        float swirl = 0.3f * fast_sin(time * 2.0f + p.posY * 4.0f);
        p.velX += swirl * dt;
        // Spinta verso l’alto
        p.velY += 0.8f * dt;
        // Resistenza
        p.velX *= 0.985f;
        p.velY *= 0.992f;
        // Posizione
        p.posX += p.velX * dt;
        p.posY += p.velY * dt;
        // Lifetime
        p.lifetime -= dt;
        p.is_dead = 0; // viva 
    }
    if (p.lifetime <= 0.0f) {
        p.is_dead = 1;
    }
    particles[idx] = p;
}

__global__ void findDeadParticlesKernel(
    const Particle* __restrict__ particles,
    int* __restrict__ dead_indices,
    int* __restrict__ dead_count,
    int numParticles,
    int gridWidth) // <-- AGGIUNGI gridWidth!
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = gy * gridWidth + gx; // <-- USA LA FORMULA 2D!
    if (idx >= numParticles) return;

    if (particles[idx].is_dead) {
        int list_idx = atomicAdd(dead_count, 1);
        dead_indices[list_idx] = idx;
    }
}

// Kernel B: respawn SOLO particelle morte – no atomics, nessuna sincronizzazione intra-blocco
__global__ void respawnDeadKernel(
    Particle* __restrict__ particles,
    const int* __restrict__ dead_indices,
    curandState* __restrict__ randStates,
    int num_to_respawn)
{
    // Usa un indice 1D perché viene lanciato come griglia 1D
    int idx_in_list = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx_in_list >= num_to_respawn) return;

    // Prendi l'indice reale della particella dalla lista compatta
    int particle_idx = dead_indices[idx_in_list];

    // Carica lo stato della particella e del generatore casuale
    Particle p = particles[particle_idx];

    // --- Logica di Respawn ---
    p.posX = (random_float(p.rand_state) - 0.5f) * 0.8f;
    p.posY = -0.8f + random_float(p.rand_state) * 0.05f;
    p.posZ = 0.0f;

    p.velX = (random_float(p.rand_state) - 0.5f) * 0.8f;
    p.velY = random_float(p.rand_state) * 1.2f + 1.0f;
    p.velZ = 0.0f;

    p.lifetime = 0.8f + random_float(p.rand_state) * 1.2f;

    p.is_dead = 0;

    // Scrivi i nuovi dati
    particles[particle_idx] = p;
}

#endif


const char* vertexShaderSource = "#version 330 core\n"
"layout (location = 0) in vec3 aPos;\n"
"layout (location = 2) in float aLifetime; // <-- NUOVO: Riceve il lifetime\n"
"\n"
"uniform mat4 projection;\n"
"\n"
"void main() {\n"
"   gl_Position = projection * vec4(aPos, 1.0);\n"
"\n"
"   // Calcolo dinamico della dimensione\n"
"   float maxLifetime = 2.0; // Valore simile a quello nella rinascita (0.8 + 1.2)\n"
"   float sizeFactor = aLifetime / maxLifetime;\n"
"   gl_PointSize = 1.0 + sizeFactor * 5.0; // Dimensione da 1 a 6 pixel\n"
"}\0";


const char* fragmentShaderSource = "#version 330 core\n"
"out vec4 FragColor;\n"
"uniform float time;\n"
"\n"
// Funzione pseudo-random animata
"float rand(vec2 co) {\n"
"    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) + time * 0.5);\n"
"}\n"
"\n"
"void main() {\n"
"    // Distanza dal centro del point sprite\n"
"    float dist = distance(gl_PointCoord, vec2(0.5));\n"
"    float radialAlpha = 1.0 - pow(dist * 2.0, 2.0);\n"
"    if (radialAlpha < 0.05) discard;\n"
"\n"
"    // Altezza normalizzata (viewport = 800 px)\n"
"    float t = gl_FragCoord.y / 800.0;\n"
"\n"
"    // Colori del fuoco\n"
"    vec3 baseColor = vec3(1.0, 0.7, 0.6);\n"
"    vec3 midColor  = vec3(1.0, 0.5, 0.0);\n"
"    vec3 topColor  = vec3(1.0, 0.0, 0.0);\n"
"\n"
"    // Interpolazione verticale\n"
"    vec3 fireColor = mix(baseColor, midColor, t);\n"
"    fireColor = mix(fireColor, topColor, t * t);\n"
"\n"
"    // Rumore animato (scintillio)\n"
"    float noise = rand(gl_FragCoord.xy * 0.1);\n"
"    fireColor += noise * 0.1;\n"
"\n"
"    // ---- ALPHA ----\n"
"    float heightAlpha = 1.0 - t;   // più alto = più trasparente\n"
"    float alpha = radialAlpha * heightAlpha;\n"
"\n"
"    FragColor = vec4(fireColor, alpha);\n"
"}\n\0";

static const char* kVertexShader = R"(#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 2) in float aLifetime;
uniform mat4 projection;
void main(){
gl_Position = projection * vec4(aPos, 1.0);
float maxLifetime = 2.0;
float sizeFactor = aLifetime / maxLifetime;
gl_PointSize = 1.0 + sizeFactor * 5.0;
}
)";


static const char* kFragmentShader = R"(#version 330 core
out vec4 FragColor;
uniform float time;
float rand(vec2 co){ return fract(sin(dot(co.xy, vec2(12.9898,78.233))) + time * 0.5); }
void main(){
float dist = distance(gl_PointCoord, vec2(0.5));
float radialAlpha = 1.0 - pow(dist * 2.0, 2.0);
if (radialAlpha < 0.05) discard;
float t = gl_FragCoord.y / 800.0;
vec3 baseColor = vec3(1.0, 0.7, 0.6);
vec3 midColor = vec3(1.0, 0.5, 0.0);
vec3 topColor = vec3(1.0, 0.0, 0.0);
vec3 fireColor = mix(baseColor, midColor, t);
fireColor = mix(fireColor, topColor, t * t);
float noise = rand(gl_FragCoord.xy * 0.1);
fireColor += noise * 0.1;
float heightAlpha = 1.0 - t;
float alpha = radialAlpha * heightAlpha;
FragColor = vec4(fireColor, alpha);
}
)";

static GLuint compileShader(GLenum type, const char* src) {
    GLuint id = glCreateShader(type);
    glShaderSource(id, 1, &src, NULL);
    glCompileShader(id);
    GLint ok = 0; glGetShaderiv(id, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[4096]; GLsizei len = 0; glGetShaderInfoLog(id, 4096, &len, log); fprintf(stderr, "Shader compile error: %s\n", log); }
    return id;
}


static GLuint buildProgram(const char* vs, const char* fs) {
    GLuint v = compileShader(GL_VERTEX_SHADER, vs);
    GLuint f = compileShader(GL_FRAGMENT_SHADER, fs);
    GLuint p = glCreateProgram();
    glAttachShader(p, v); glAttachShader(p, f);
    glLinkProgram(p);
    glDeleteShader(v); glDeleteShader(f);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char log[4096]; GLsizei len = 0; glGetProgramInfoLog(p, 4096, &len, log); fprintf(stderr, "Program link error: %s\n", log); }
    return p;
}


int main(void)
{

    // INIT RENDERING
    // Init glfw
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(800, 800, "Simulazione Fuoco", NULL, NULL);
    glfwMakeContextCurrent(window);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

    // Compilazione degli shader 
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // Dimensione fuoco
    float world_left = -1.0f;
    float world_right = 1.0f;
    float world_bottom = 0.0f;
    float world_top = 4.0f;

    // Crezione matrice di proiezione ortografica con GLM
    glm::mat4 projection = glm::ortho(world_left, world_right, world_bottom, world_top, -1.0f, 1.0f);
    int projectionLoc = glGetUniformLocation(shaderProgram, "projection");


    // INIZIALIZZAZIONE
    const int NUM_PARTICLES = 1048576;
    size_t size = NUM_PARTICLES * sizeof(Particle);
    size_t vertices_size = NUM_PARTICLES * sizeof(ParticleVertex);

    GLuint VBO, VAO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);



#if defined(SEQUENTIAL_BASELINE)

    printf("Esecuzione in modalità: SEQUENTIAL_BASELINE (CPU)\n");

    // Allocazione memoria particelle
    Particle* h_particles = (Particle*)malloc(size);
    ParticleVertex* h_vertices = (ParticleVertex*)malloc(vertices_size);

    // Inizializzazione particelle
    initializeParticles(h_particles, NUM_PARTICLES);

    // Pulizia vertici
    memset(h_vertices, 0, vertices_size);

    // Misurazione del tempo per un singolo aggiornamento come da linee guida
    clock_t start_cpu = clock();

    updateParticlesCPU(h_particles, NUM_PARTICLES, 0.016f, 0.0f);

    clock_t stop_cpu = clock();

    double duration_cpu_us = ((double)(stop_cpu - start_cpu) / CLOCKS_PER_SEC) * 1000000.0;
    printf("Tempo di esecuzione CPU per un update: %.0f microsecondi\n", duration_cpu_us);

    // Setup di VBO e VAO per il rendering
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices_size, NULL, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(ParticleVertex), (void*)0);
    glEnableVertexAttribArray(0);

    // Loop di rendering principale
    while (!glfwWindowShouldClose(window)) {
        float t = (float)glfwGetTime();

        // Aggiorna stato particelle su CPU
        updateParticlesCPU(h_particles, NUM_PARTICLES, 0.016f, t);

        // Mappa i dati delle particelle aggiornate ai vertici per il rendering
        for (int i = 0; i < NUM_PARTICLES; i++) {
            h_vertices[i].x = h_particles[i].posX;
            h_vertices[i].y = h_particles[i].posY;
            h_vertices[i].z = h_particles[i].posZ;
        }

        // Carica i dati dei vertici nel VBO
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, vertices_size, h_vertices);

        // Disegna
        glClearColor(0.0f, 0.0f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);

        // Attivazione shader
        glUseProgram(shaderProgram);
        glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, &projection[0][0]);
        glBindVertexArray(VAO);
        glDrawArrays(GL_POINTS, 0, NUM_PARTICLES);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Liberazione risorse
    free(h_particles);
    free(h_vertices);

#elif defined(GPU_NAIVE_PARALLEL)

    // Definizione dimensione thread block e grid
    const int BLOCK_SIZE = 256;
    int gridSize = (NUM_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Allocazione memoria particelle
    Particle* h_particles = (Particle*)malloc(size);

    // Inizializzazione particelle
    initializeParticles(h_particles, NUM_PARTICLES);

    Particle* d_particles;
    curandState* d_randStates;
    cudaMalloc((void**)&d_particles, size);
    cudaMalloc((void**)&d_randStates, NUM_PARTICLES * sizeof(curandState));
    cudaMemcpy(d_particles, h_particles, size, cudaMemcpyHostToDevice);
    free(h_particles);

    // Eseguito solo una volta
    initCurandKernel << <gridSize, BLOCK_SIZE >> > (d_randStates, NUM_PARTICLES, time(NULL));

    //OpenGL
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices_size, NULL, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(ParticleVertex), (void*)0);
    glEnableVertexAttribArray(0);

    // Comunicazione a cuda per accesso a vbo
    struct cudaGraphicsResource* cuda_vbo_resource;
    cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, VBO, cudaGraphicsRegisterFlagsNone);

    while (!glfwWindowShouldClose(window))
    {
        // Puntatore vbo per cuda, blocco modifiche su vbo (da OpenGL)
        ParticleVertex* d_vbo_ptr;
        cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
        size_t num_bytes;
        cudaGraphicsResourceGetMappedPointer((void**)&d_vbo_ptr, &num_bytes, cuda_vbo_resource);

        float t = glfwGetTime();

        //Misurazione del tempo del kernel
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);

        // AGGIORNAMENTO STATO PARTICELLE
        updateParticlesKernel << <gridSize, BLOCK_SIZE >> > (d_particles, d_vbo_ptr, NUM_PARTICLES, 0.016f, d_randStates, t);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        static int frameCount = 0;
        if (frameCount % 60 == 0) {
            printf("Tempo di esecuzione GPU per un update: %.4f ms\n", milliseconds);
        }
        frameCount++;

        // DISEGNO
        // Restituzione vbo a OpenGL per disegnare
        cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);

        glClearColor(0.0f, 0.0f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);

        // Attivazione shader
        glUseProgram(shaderProgram);
        glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, &projection[0][0]);
        glBindVertexArray(VAO);
        glDrawArrays(GL_POINTS, 0, NUM_PARTICLES);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Rilascio risorse
    cudaFree(d_particles);
    cudaFree(d_randStates);
#endif

    glDeleteProgram(shaderProgram);
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glfwTerminate();

    return 0;
}

int main() {

    srand(time(NULL));
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(800, 800, "Fire Simulation [DEBUG POS_X]", NULL, NULL);
    glfwMakeContextCurrent(window);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
    cudaSetDevice(0);

    // Assicurati di usare gli shader di debug per posX che ti ho dato
    GLuint shaderProgram = buildProgram(kVertexShader, kFragmentShader);

    glm::mat4 projection = glm::ortho(-1.0f, 1.0f, 0.0f, 4.0f, -1.0f, 1.0f);
    int projectionLoc = glGetUniformLocation(shaderProgram, "projection");

    const int NUM_PARTICLES = 1048576;
    size_t vertices_size = NUM_PARTICLES * sizeof(ParticleVertex);
    size_t array_size = NUM_PARTICLES * sizeof(float);
    size_t rand_array_size = NUM_PARTICLES * sizeof(unsigned int);

    // --- SETUP VBO/VAO AGGIORNATO PER DEBUG POS_X ---
    GLuint VBO, VAO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices_size, NULL, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(ParticleVertex), (void*)0);
    glEnableVertexAttribArray(0);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glBindVertexArray(0);
    // --- FINE SETUP VBO/VAO ---

    struct cudaGraphicsResource* cuda_vbo_resource;
    cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, VBO, cudaGraphicsRegisterFlagsNone);

    setupSinLut();
    cudaDeviceSynchronize();

    ParticlesSoA h_soa, d_soa;
    // Allocazione Host
    h_soa.posX = (float*)malloc(array_size); h_soa.posY = (float*)malloc(array_size);// h_soa.posZ = (float*)malloc(array_size);
    h_soa.velX = (float*)malloc(array_size); h_soa.velY = (float*)malloc(array_size);// h_soa.velZ = (float*)malloc(array_size);
    h_soa.lifetime = (float*)malloc(array_size); h_soa.rand_state = (unsigned int*)malloc(rand_array_size);
    // Allocazione Device
    cudaMalloc(&d_soa.posX, array_size); cudaMalloc(&d_soa.posY, array_size); //cudaMalloc(&d_soa.posZ, array_size);
    cudaMalloc(&d_soa.velX, array_size); cudaMalloc(&d_soa.velY, array_size); //cudaMalloc(&d_soa.velZ, array_size);
    cudaMalloc(&d_soa.lifetime, array_size); cudaMalloc(&d_soa.rand_state, rand_array_size);

    initializeParticles_SoA(h_soa, NUM_PARTICLES);

    cudaMemcpy(d_soa.posX, h_soa.posX, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_soa.posY, h_soa.posY, array_size, cudaMemcpyHostToDevice);
    //cudaMemcpy(d_soa.posZ, h_soa.posZ, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_soa.velX, h_soa.velX, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_soa.velY, h_soa.velY, array_size, cudaMemcpyHostToDevice);
    //cudaMemcpy(d_soa.velZ, h_soa.velZ, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_soa.lifetime, h_soa.lifetime, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_soa.rand_state, h_soa.rand_state, rand_array_size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    cleanup_SoA_Host(h_soa); // Libera la memoria CPU, non serve più

    dim3 blockDim(512);
    dim3 gridDim((NUM_PARTICLES + blockDim.x - 1) / blockDim.x);
    float lastTime = 0.0f;


    while (!glfwWindowShouldClose(window)) {
        float currentTime = (float)glfwGetTime();
        float dt = 0.016f;

        glfwPollEvents();

        ParticleVertex* d_vbo_ptr = nullptr;
        cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
        size_t num_bytes;
        cudaGraphicsResourceGetMappedPointer((void**)&d_vbo_ptr, &num_bytes, cuda_vbo_resource);

        fireKernel_SoA_FINAL << <gridDim, blockDim >> > (
            d_soa.posX, d_soa.posY,
            //d_soa.posZ,
            d_soa.velX, d_soa.velY,
            //d_soa.velZ,
            d_soa.lifetime, d_soa.rand_state,
            NUM_PARTICLES, dt, currentTime
            );
        updateVBOKernel_SoA << <gridDim, blockDim >> > (
            d_soa.posX, d_soa.posY,
            //d_soa.posZ,
            d_soa.lifetime,
            d_vbo_ptr, NUM_PARTICLES
            );



        cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);

        // --- Rendering OpenGL ---
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);

        glUseProgram(shaderProgram);
        glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, &projection[0][0]);

        glBindVertexArray(VAO);
        glDrawArrays(GL_POINTS, 0, NUM_PARTICLES);
        glBindVertexArray(0); // Buona pratica

        glfwSwapBuffers(window);
    }

    // Cleanup
    cudaDeviceSynchronize();

    cudaGraphicsUnregisterResource(cuda_vbo_resource);
    cleanup_SoA_Device(d_soa);

    glDeleteProgram(shaderProgram);
    glDeleteBuffers(1, &VBO);
    glDeleteVertexArrays(1, &VAO);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
