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

#define BLOCK_SIZE 1024
#define MAX_NEIGHBORS 32      // Numero massimo di vicini in shared memory
#define CELL_SIZE 1.0f        // Dimensione di una cella della griglia
#define TILE_WIDTH 32
#define TILE_HEIGHT 16

//#define SEQUENTIAL_BASELINE
//#define GPU_NAIVE_PARALLEL
//#define OPTIMIZED_DIVERGENCE_VERSION
#define OPTIMIZED_DIVERGENCE_2
//#define SHARED_MEM_VERSION

// Stato particella
typedef struct {
    float posX, posY, posZ;
    float velX, velY, velZ;
    float lifetime;
    float age;
} Particle;

// Dati per rendering particella
typedef struct {
    float x, y, z;
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
__global__ void updateAliveKernel(
    Particle* particles,
    int* __restrict__ deadFlags, // 1 = morta, 0 = viva
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


    if (p.lifetime > 0.0f) {
        // Forza di convergenza
        p.velX -= p.posX * 1.2f * dt;
        // Turbolenza
        float turbulence = __sinf(p.posY * 3.0f + time * 1.5f + p.posX * 2.0f)
            + __cosf(p.posY * 5.0f + time * 2.5f);
        p.velX += turbulence * 0.4f * dt;
        // Swirl
        float swirl = 0.3f * __sinf(time * 2.0f + p.posY * 4.0f);
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


        particles[idx] = p;
        deadFlags[idx] = 0; // viva
    }
    else {
        // Non aggiorno – sarà respawnata in B
        deadFlags[idx] = 1; // morta
    }
}

// Kernel B: respawn SOLO particelle morte – no atomics, nessuna sincronizzazione intra-blocco
__global__ void respawnDeadKernel(
    Particle* particles,
    int* __restrict__ deadFlags,
    curandState* __restrict__ randStates,
    int numParticles,
    int gridWidth)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = gy * gridWidth + gx;
    if (idx >= numParticles) return;

    if (deadFlags[idx]) {
        Particle p = particles[idx];
        curandState localState = randStates[idx];

        // *** THIS IS THE MISSING PART ***
        // Copy-pasted from your original respawn logic:
        p.posX = (curand_uniform(&localState) - 0.5f) * 0.8f;
        p.posY = -0.8f + curand_uniform(&localState) * 0.05f;
        p.posZ = 0.0f;

        p.velX = (curand_uniform(&localState) - 0.5f) * 0.8f;
        p.velY = curand_uniform(&localState) * 1.2f + 1.0f;
        p.velZ = 0.0f;

        p.lifetime = 0.8f + curand_uniform(&localState) * 1.2f;

        particles[idx] = p; // Update global memory
        randStates[idx] = localState; // Update global randState
    }
}

#endif

__global__ void updateVBOKernel(const Particle* __restrict__ particles, ParticleVertex* __restrict__ vbo_ptr, int numParticles, int gridWidth)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = gy * gridWidth + gx;
    if (idx >= numParticles) return;

    Particle p = particles[idx];
    vbo_ptr[idx].x = p.posX;
    vbo_ptr[idx].y = p.posY;
    vbo_ptr[idx].z = p.posZ;
    vbo_ptr[idx].lifetime = p.lifetime;
}

#ifdef SHARED_MEM_VERSION
__global__ void updateParticlesKernel(Particle* particles, ParticleVertex* vbo_ptr,
    int numParticles, float dt, curandState* randStates, float time)
{
    static auto lastTime = std::chrono::high_resolution_clock::now();
    auto now = std::chrono::high_resolution_clock::now();
    float deltaTime = std::chrono::duration<float>(now - lastTime).count();
    lastTime = now;
    __shared__ Particle sharedParticles[MAX_NEIGHBORS];
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= numParticles) return;

        Particle p = particles[idx];

        // --- 1. Copia vicini nella shared memory ---
        int neighborCount = 0;
        for (int i = 0; i < numParticles && neighborCount < MAX_NEIGHBORS; i++) {
            float dx = particles[i].posX - p.posX;
            float dy = particles[i].posY - p.posY;
            float dz = particles[i].posZ - p.posZ;
            float dist2 = dx * dx + dy * dy + dz * dz;
            if (dist2 < CELL_SIZE * CELL_SIZE) {
                sharedParticles[neighborCount++] = particles[i];
            }
        }
        __syncthreads();

        // --- 2. Calcola velocità media locale ---
        float velSumX = 0.0f, velSumY = 0.0f, velSumZ = 0.0f;
        for (int i = 0; i < neighborCount; i++) {
            velSumX += sharedParticles[i].velX;
            velSumY += sharedParticles[i].velY;
            velSumZ += sharedParticles[i].velZ;
        }
        float invCount = 1.0f / (float)neighborCount;
        float velAvgX = velSumX * invCount;
        float velAvgY = velSumY * invCount;
        float velAvgZ = velSumZ * invCount;

        // --- 3. Aggiungi turbolenza coerente ---
        float noiseX = sinf(p.posX * 10.0f + p.age * 5.0f);
        float noiseY = cosf(p.posY * 10.0f + p.age * 5.0f);
        float noiseZ = sinf(p.posZ * 10.0f + p.age * 5.0f);
        float turbulenceScale = 0.1f;

        // --- 4. Aggiorna velocità e posizione ---
        p.velX = velAvgX + noiseX * turbulenceScale;
        p.velY = velAvgY + noiseY * turbulenceScale;
        p.velZ = velAvgZ + noiseZ * turbulenceScale;

        p.posX += p.velX * deltaTime;
        p.posY += p.velY * deltaTime;
        p.posZ += p.velZ * deltaTime;

        // --- 5. Aggiorna età e rinascita ---
        p.age += deltaTime;
        if (p.age > p.lifetime) {
            p.posX = 0.0f;          // nuova posizione di emissione
            p.posY = 0.0f;
            p.posZ = 0.0f;
            p.velX = 0.0f;          // nuova velocità iniziale
            p.velY = 1.0f;
            p.velZ = 0.0f;
            p.age = 0.0f;
            p.lifetime = 1.0f + 0.5f * (float)(idx % 10); // leggero random
        }

        // --- 6. Scrivi nella memoria globale ---

particles[idx] = p;

// Scrivo direttamente su VBO aggiornamento dati di posizione e colore delle particelle
vbo_ptr[idx].x = p.posX;
vbo_ptr[idx].y = p.posY;
vbo_ptr[idx].z = 0.0f;

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

out float vPosY; // Output particle's Y position to fragment shader

uniform mat4 projection;

void main(){
    gl_Position = projection * vec4(aPos, 1.0);
    float maxLifetime = 2.0;
    float sizeFactor = aLifetime / maxLifetime;
    gl_PointSize = 1.0 + sizeFactor * 5.0;

    // Pass the particle's actual Y position to the fragment shader
    vPosY = aPos.y;
}
)";


static const char* kFragmentShader = R"(#version 330 core
in float vPosY; // Receive particle's Y position from vertex shader
out vec4 FragColor;
uniform float time;

float rand(vec2 co){ return fract(sin(dot(co.xy, vec2(12.9898,78.233))) + time * 0.5); }

void main(){
    float dist = distance(gl_PointCoord, vec2(0.5));
    float radialAlpha = 1.0 - pow(dist * 2.0, 2.0);
    if (radialAlpha < 0.05) discard;

    // Normalize particle's Y position within its world bounds (0.0 to 4.0)
    // Your world_bottom is 0.0f, world_top is 4.0f
    float normalizedWorldY = (vPosY - 0.0) / (4.0 - 0.0);
    float t = clamp(normalizedWorldY, 0.0, 1.0); // Ensure t is between 0 and 1

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

int main(void) {

    //________________________________INIT RENDERING______________________________________________
    // Init glfw
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* window = glfwCreateWindow(800, 800, "Simulazione Fuoco", NULL, NULL);
    glfwMakeContextCurrent(window);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

    GLuint program = buildProgram(kVertexShader, kFragmentShader);

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

    GLuint VBO, VAO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    //___________________________________________________________________________________________________

    ///______________________________INIZIALIZZAZIONE E ALLOCAZIONE MEMORIA (HOST)________________________________
    const int NUM_PARTICLES = 1048576;
    size_t size = NUM_PARTICLES * sizeof(Particle);
    size_t vertices_size = NUM_PARTICLES * sizeof(ParticleVertex);

#if defined(SEQUENTIAL_BASELINE)

    // Allocazione memoria particelle (host)
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

#elif defined(GPU_NAIVE_PARALLEL) || defined(OPTIMIZED_DIVERGENCE_VERSION) || defined(OPTIMIZED_DIVERGENCE_2)
    
    // Allocazione memoria particelle (host)
    Particle* h_particles = (Particle*)malloc(size);

    // Inizializzazione particelle
    initializeParticles(h_particles, NUM_PARTICLES);

    // Allocazione memoria particelle (device)
    Particle* d_particles;
    curandState* d_randStates;
#ifdef OPTIMIZED_DIVERGENCE_2
    int* d_deadFlags = nullptr;
    cudaMalloc((void**)&d_deadFlags, NUM_PARTICLES * sizeof(int));
#endif
    cudaMalloc((void**)&d_particles, size);
    cudaMalloc((void**)&d_randStates, NUM_PARTICLES * sizeof(curandState));

    // Trasferimento particelle inizializzate (Host->Device)
    cudaMemcpy(d_particles, h_particles, size, cudaMemcpyHostToDevice);
    free(h_particles); 
 
    // Definizione dimensione grid e thread block 1D (per initCurandKernel)
    dim3 gridDim1D((NUM_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
    dim3 blockDim1D(BLOCK_SIZE, 1, 1);

    // Esecuzione kernel di inizializzazione
    initCurandKernel << <gridDim1D, blockDim1D >> > (d_randStates, NUM_PARTICLES, time(NULL));

    // Definizione dimensione grid e thread block 2D (per updateParticlesKernel)
    // NUM_PARTICLES -> dataSizeX * dataSizeY = 1,048,576 particelle
    const int dataSizeX = 1024;
    const int dataSizeY = 1024;

    dim3 blockDim(32, 16, 1);
    dim3 gridDim(
        (dataSizeX + blockDim.x - 1) / blockDim.x,
        (dataSizeY + blockDim.y - 1) / blockDim.y, 1);
    int gridWidth = blockDim.x * gridDim.x;
    //OpenGL
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices_size, NULL, GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(ParticleVertex), (void*)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(ParticleVertex), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(ParticleVertex), (void*)(7 * sizeof(float)));
    glEnableVertexAttribArray(2);

    glEnable(GL_PROGRAM_POINT_SIZE);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

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
        
#if defined(GPU_NAIVE_PARALLEL) || defined(OPTIMIZED_DIVERGENCE_VERSION)
        // Esecuzione kernel aggiornamento
        updateParticlesKernel << <gridDim, blockDim >> > (d_particles, d_vbo_ptr, NUM_PARTICLES, 0.016f, d_randStates, t, gridWidth);

        updateVBOKernel << <gridDim, blockDim >> > (d_particles, d_vbo_ptr, NUM_PARTICLES, gridWidth);
#elif defined(OPTIMIZED_DIVERGENCE_2)
        // A) Update only alive (mark deads)
        updateAliveKernel << <gridDim, blockDim >> > (
            d_particles, d_deadFlags, NUM_PARTICLES, 0.016f, t, gridWidth);


        // B) Respawn only dead (no atomics)
        respawnDeadKernel << <gridDim, blockDim >> > (
            d_particles, d_deadFlags, d_randStates, NUM_PARTICLES, gridWidth);


        // C) Copy to VBO
        updateVBOKernel << <gridDim, blockDim >> > (
            d_particles, d_vbo_ptr, NUM_PARTICLES, gridWidth);
#endif
        //Calcolo del tempo per speedup
        cudaEventRecord(stop);
        cudaEventSynchronize(stop); //Sincronizzazione, comando bloccante

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
        
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
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
    cudaFree(d_particles);
    cudaFree(d_randStates);

#endif



    glDeleteProgram(shaderProgram);
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glfwTerminate();

    return 0;
}

