#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <stddef.h> 
#include "fast_math.cuh"

#define SUB_STEPS 8

#define INTERACTION_RADIUS 0.55f
#define CELL_SIZE INTERACTION_RADIUS
#define MAX_PARTICLES_PER_CELL 1024
#define TURB_PERIOD 5.0f
// =================================================================================
// SEZIONE 1: STRUTTURE DATI SoA E HELPER
// =================================================================================

struct ParticlesSoA {
    float* posX; float* posY; //float* posZ;
    float* velX; float* velY; //float* velZ;
    float* lifetime; float* turbulence;
    unsigned int* rand_state;
};

struct ParticleVertex {
    float x, y, z, lifetime, turbulence;
};

void initializeParticles_SoA(ParticlesSoA& p, int numParticles) {
    srand((unsigned)time(NULL));
    for (int i = 0; i < numParticles; ++i) {
        p.posX[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.15f; // <--- non zero
        p.posY[i] = -0.8f + (rand() / (float)RAND_MAX) * 0.1f;   // <--- non zero
        p.velX[i] = 0.0f;
        p.velY[i] = 0.0f;
        p.lifetime[i] = (rand() / (float)RAND_MAX) * 2.0f + 0.5f; // 0.5..2.5 (coerente con shader)
        p.rand_state[i] = rand() + 1u;
    }
}

void cleanup_SoA_Host(ParticlesSoA& p) {
    free(p.posX); free(p.posY); //free(p.posZ);
    free(p.velX); free(p.velY); //free(p.velZ);
    free(p.lifetime); free(p.turbulence);
    free(p.rand_state);
}

void cleanup_SoA_Device(ParticlesSoA& p) {
    cudaFree(p.posX); cudaFree(p.posY); //cudaFree(p.posZ);
    cudaFree(p.velX); cudaFree(p.velY); //cudaFree(p.velZ);
    cudaFree(p.lifetime); cudaFree(p.turbulence);
    cudaFree(p.rand_state);
}
void computeParticleCells(float* posX, float* posY, int* particleCell,
    int numParticles, int numCellsX, int numCellsY)
{
    for (int i = 0; i < numParticles; i++) {
        int cellX = min(max(int((posX[i] + 1.0f) / CELL_SIZE), 0), numCellsX - 1);
        int cellY = min(max(int((posY[i] + 1.0f) / CELL_SIZE), 0), numCellsY - 1);
        particleCell[i] = cellY * numCellsX + cellX;
    }
}
void computeCellStartCount(int* particleCell, int numParticles,
    int* cellStart, int* cellCount, int numCells)
{
    for (int i = 0; i < numCells; i++) {
        cellStart[i] = -1;
        cellCount[i] = 0;
    }
    for (int i = 0; i < numParticles; i++) {
        int c = particleCell[i];
        if (cellStart[c] == -1) cellStart[c] = i;
        cellCount[c]++;
    }
}



#define SUB_STEPS 8
#define MAX_PARTICLES_PER_CELL 128
#define CELL_SIZE 0.05f

__global__ void fireKernel_SoA_Cells(
    float* p_posX, float* p_posY,
    float* p_velX, float* p_velY,
    float* p_lifetime, float* p_turbulence,
    unsigned int* p_rand_state,
    int numParticles,
    float dt, float time,
    int numCellsX, int numCellsY,
    int* cellStart, int* cellCount, int* particleCell)
{
    int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (gidx >= numParticles) return;

    // Carico i dati in variabili locali (registri) per i calcoli iniziali
    float l_posX = p_posX[gidx];
    float l_posY = p_posY[gidx];
    float l_velX = p_velX[gidx];
    float l_velY = p_velY[gidx];
    float l_life = p_lifetime[gidx];
    float l_turb = p_turbulence[gidx];
    unsigned int local_rand = p_rand_state[gidx];

    const float sub_dt = dt / SUB_STEPS;

    // --- Passo 1: Fisica di base (sub-steps) ---
#pragma unroll
    for (int i = 0; i < SUB_STEPS; ++i) {
        float t = time + i * sub_dt;
        if (l_life <= 0.0f) {
            // Reset delle particelle
            l_posX = (random_float(local_rand) - 0.5f) * 0.2f;
            l_posY = -0.9f + random_float(local_rand) * 0.15f;
            l_velX = (random_float(local_rand) - 0.5f) * 1.0f;
            l_velY = 4.0f + random_float(local_rand) * 2.0f;
            l_life = 2.5f + random_float(local_rand) * 1.5f;
            l_turb = 0.0f;
        }
        else {
            // Fisica standard
            l_velX -= l_posX * 3.0f * sub_dt;
            l_velY += 2.0f * sub_dt;
            float base_turb = fast_sin(fmaf(l_posY, 3.0f, fmaf(t, 2.0f, l_posX * 2.0f)))
                + fast_cos(fmaf(l_posY, 5.0f, t * 2.5f));
            l_velX += base_turb * 0.1f * sub_dt;
            l_velX *= 0.985f; l_velY *= 0.992f;
            l_posX += l_velX * sub_dt;
            l_posY += l_velY * sub_dt;
            l_life -= sub_dt;
        }
    }

    // --- Passo 2: Interazione basata sulla cella ---
    // Trovo la cella dopo l'aggiornamento
    int cellX = min(max(int((l_posX + 1.0f) / CELL_SIZE), 0), numCellsX - 1);
    int cellY = min(max(int((l_posY + 1.0f) / CELL_SIZE), 0), numCellsY - 1);
    int cellIdx = cellY * numCellsX + cellX;

    int start = cellStart[cellIdx];
    int count = cellCount[cellIdx];
    if (count > MAX_PARTICLES_PER_CELL) count = MAX_PARTICLES_PER_CELL;

    extern __shared__ float s_data[];
    float* s_posX = s_data;
    float* s_posY = s_data + MAX_PARTICLES_PER_CELL;

    // Carico i dati della cella nella shared memory
    if (threadIdx.x < count) {
        s_posX[threadIdx.x] = p_posX[start + threadIdx.x];
        s_posY[threadIdx.x] = p_posY[start + threadIdx.x];
    }
    __syncthreads();

    // --- Passo 3: Calcolo delle interazioni ---
    if (gidx < numParticles) { // Evita accessi fuori limite per i thread in eccesso
        for (int i = 0; i < count; ++i) {
            float other_posX = s_posX[i];
            float other_posY = s_posY[i];

            float dx = l_posX - other_posX;
            float dy = l_posY - other_posY;
            float dist_sq = dx * dx + dy * dy;

            if (dist_sq < 0.05f * 0.05f) {
                float inv_dist = rsqrtf(dist_sq);
                float force = 0.005f * inv_dist;
                l_velX += force * dx;
                l_velY += force * dy;
            }
        }
    }

    // --- Passo 4: Aggiornamento finale e scrittura globale ---
    p_posX[gidx] = l_posX;
    p_posY[gidx] = l_posY;
    p_velX[gidx] = l_velX;
    p_velY[gidx] = l_velY;
    p_lifetime[gidx] = l_life;
    p_turbulence[gidx] = l_turb;
    p_rand_state[gidx] = local_rand;
}


__global__ void updateVBOKernel_SoA(
    float* p_posX, float* p_posY,
    float* p_lifetime, float* p_turbulence,
    ParticleVertex* vbo_ptr, int numParticles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;
    vbo_ptr[idx].x = p_posX[idx];
    vbo_ptr[idx].y = p_posY[idx];
    vbo_ptr[idx].lifetime = p_lifetime[idx];
    vbo_ptr[idx].turbulence = p_turbulence[idx];

}

// =================================================================================
// SEZIONE 3: SHADER E MAIN
// =================================================================================
static const char* kVertexShader = R"(#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in float aLifetime;
layout(location = 2) in float aTurbulence;
uniform mat4 projection;


out float vLifetime;
out float vTurbulence;
    void main(){
       gl_Position = projection * vec4(aPos, 1.0);
       vLifetime = aLifetime;
       vTurbulence = aTurbulence;

        //vLifetime = aPosLifetime.w;
        float sizeFactor = clamp(vLifetime / 2.5, 0.0, 1.0);
        gl_PointSize = 1.0 + sizeFactor * 10.0; 
    }
)";
static const char* kFragmentShader = R"(#version 330 core
in float vLifetime;
in float vTurbulence; // aggiungi questo attributo dal VBO
out vec4 FragColor;

void main(){
    float dist = distance(gl_PointCoord, vec2(0.5));
    float radialAlpha = 1.0 - pow(dist * 2.0, 4.0);
    if (radialAlpha < 0.0) discard;

    if (vTurbulence >= 1.0) {
        // Tutte le particelle blu
        FragColor = vec4(0.0, 0.3, 1.0, radialAlpha); 
    } 
    else {
        float t = clamp(vLifetime / 3.0, 0.0, 1.0);

        vec3 coolColor = vec3(0.5, 0.1, 0.0);
        vec3 midColor  = vec3(0.9, 0.35, 0.0);
        vec3 hotColor  = vec3(1.0, 0.6, 0.1);
        vec3 coreColor = vec3(1.0, 0.9, 0.7);

        vec3 color;
        if (t > 0.9) {
            color = mix(hotColor, coreColor, (t - 0.9) / 0.1); 
        } else if (t > 0.6) {
            color = mix(midColor, hotColor, (t - 0.6) / 0.3); 
        } else {
            color = mix(coolColor, midColor, t / 0.6); 
        }

        float finalAlpha = radialAlpha * pow(t, 0.8) * 0.4;
        FragColor = vec4(color, finalAlpha);
    }
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

int main() {

    srand(time(NULL));
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(800, 800, "Fire Simulation", NULL, NULL);
    glfwMakeContextCurrent(window);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
    cudaSetDevice(0);

    GLuint shaderProgram = buildProgram(kVertexShader, kFragmentShader);

    glm::mat4 projection = glm::ortho(-1.0f, 1.0f, 0.0f, 4.0f, -1.0f, 1.0f);
    int projectionLoc = glGetUniformLocation(shaderProgram, "projection");

    const int NUM_PARTICLES = 1048576;
    size_t vertices_size = NUM_PARTICLES * sizeof(ParticleVertex);
    size_t array_size = NUM_PARTICLES * sizeof(float);
    size_t rand_array_size = NUM_PARTICLES * sizeof(unsigned int);

    GLuint VBO, VAO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices_size, NULL, GL_DYNAMIC_DRAW);

    // Attributo posizione (vec2) e lifetime (float)
    // ParticleVertex: float x, y, z, lifetime;
    // La posizione (x,y) e il lifetime (w) sono nello stesso vec4, z non usato in questo shader
    // Location 0: posizione xyz
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(ParticleVertex), (void*)0);
    glEnableVertexAttribArray(0);

    // Location 1: lifetime
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(ParticleVertex), (void*)offsetof(ParticleVertex, lifetime));
    glEnableVertexAttribArray(1);

    // Location 2: turbulence
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(ParticleVertex), (void*)offsetof(ParticleVertex, turbulence));
    glEnableVertexAttribArray(2);



    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE); // BLENDING ADDITIVO per effetto fuoco/luce
    glBindVertexArray(0);

    struct cudaGraphicsResource* cuda_vbo_resource;
    cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, VBO, cudaGraphicsRegisterFlagsNone);

    setupSinLut();
    cudaDeviceSynchronize();


    

    ParticlesSoA h_soa, d_soa;
    // Allocazione Host
    h_soa.posX = (float*)malloc(array_size); h_soa.posY = (float*)malloc(array_size);
    h_soa.velX = (float*)malloc(array_size); h_soa.velY = (float*)malloc(array_size);
    h_soa.lifetime = (float*)malloc(array_size); h_soa.turbulence = (float*)malloc(array_size);
    h_soa.rand_state = (unsigned int*)malloc(rand_array_size);
    // Allocazione Device
    cudaMalloc(&d_soa.posX, array_size); cudaMalloc(&d_soa.posY, array_size);
    cudaMalloc(&d_soa.velX, array_size); cudaMalloc(&d_soa.velY, array_size);
    cudaMalloc(&d_soa.lifetime, array_size); cudaMalloc(&d_soa.turbulence, array_size);
    cudaMalloc(&d_soa.rand_state, rand_array_size);

    initializeParticles_SoA(h_soa, NUM_PARTICLES);

    cudaMemcpy(d_soa.posX, h_soa.posX, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_soa.posY, h_soa.posY, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_soa.velX, h_soa.velX, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_soa.velY, h_soa.velY, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_soa.lifetime, h_soa.lifetime, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_soa.turbulence, h_soa.turbulence, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_soa.rand_state, h_soa.rand_state, rand_array_size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // Numero massimo di celle lungo X e Y
    int NUM_CELLS_X = 40; // dipende da INTERACTION_RADIUS e dimensioni spazio
    int NUM_CELLS_Y = 40;
    int NUM_CELLS = NUM_CELLS_X * NUM_CELLS_Y;

    // Array device
    int* d_cellStart;     // indice della prima particella in ogni cella
    int* d_cellCount;     // quante particelle ci sono in ogni cella
    int* d_particleCell;  // cella di ciascuna particella

    cudaMalloc(&d_cellStart, NUM_CELLS * sizeof(int));
    cudaMalloc(&d_cellCount, NUM_CELLS * sizeof(int));
    cudaMalloc(&d_particleCell, NUM_PARTICLES * sizeof(int));

    int* h_particleCell = (int*)malloc(NUM_PARTICLES * sizeof(int));
    int* h_cellStart = (int*)malloc(NUM_CELLS * sizeof(int));
    int* h_cellCount = (int*)malloc(NUM_CELLS * sizeof(int));

    computeParticleCells(h_soa.posX, h_soa.posY, h_particleCell, NUM_PARTICLES, NUM_CELLS_X, NUM_CELLS_Y);
    computeCellStartCount(h_particleCell, NUM_PARTICLES, h_cellStart, h_cellCount, NUM_CELLS);

    cudaMemcpy(d_particleCell, h_particleCell, NUM_PARTICLES * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cellStart, h_cellStart, NUM_CELLS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cellCount, h_cellCount, NUM_CELLS * sizeof(int), cudaMemcpyHostToDevice);


    cleanup_SoA_Host(h_soa); // Libera la memoria CPU, non serve più

    dim3 blockDim(512); // Dimensioni del blocco standard
    dim3 gridDim((NUM_PARTICLES + blockDim.x - 1) / blockDim.x);

    size_t shmem_size = MAX_PARTICLES_PER_CELL * 6 * sizeof(float); // posX,posY,velX,velY,lifetime,turb
   

    float lastTime = 0.0f;


    while (!glfwWindowShouldClose(window)) {
        float currentTime = (float)glfwGetTime();
        float dt = 0.016f; // Un deltaTime fisso per stabilità

        glfwPollEvents();

        ParticleVertex* d_vbo_ptr = nullptr;
        cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
        size_t num_bytes;
        cudaGraphicsResourceGetMappedPointer((void**)&d_vbo_ptr, &num_bytes, cuda_vbo_resource);

        // Lancio del kernel fireKernel_SoA_FINAL con la shared memory specificata
        fireKernel_SoA_Cells << <gridDim, blockDim, shmem_size >> > (
            d_soa.posX, d_soa.posY,
            d_soa.velX, d_soa.velY,
            d_soa.lifetime, d_soa.turbulence,
            d_soa.rand_state,
            NUM_PARTICLES, dt, currentTime,
            NUM_CELLS_X, NUM_CELLS_Y,
            d_cellStart, d_cellCount, d_particleCell
            );



        // Questo kernel non necessita di shared memory perché legge dalla global memory
        // e scrive direttamente nel VBO mappato, che ha un pattern di accesso coalesced.
        updateVBOKernel_SoA << <gridDim, blockDim >> > (
            d_soa.posX, d_soa.posY,
            d_soa.lifetime, d_soa.turbulence,
            d_vbo_ptr, NUM_PARTICLES
            );

        cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
        cudaDeviceSynchronize(); // Aggiunto per debugging e per assicurarsi che i kernel siano finiti

        // --- Rendering OpenGL ---
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        // Blending è già abilitato all'inizio del main

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
