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
// =================================================================================
// SEZIONE 1: STRUTTURE DATI SoA E HELPER
// =================================================================================

struct ParticlesSoA {
    float* posX; float* posY; //float* posZ;
    float* velX; float* velY; //float* velZ;
    float* lifetime;
    unsigned int* rand_state;
};

struct ParticleVertex {
    float x, y, z, lifetime;
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
    free(p.lifetime); free(p.rand_state);
}

void cleanup_SoA_Device(ParticlesSoA& p) {
    cudaFree(p.posX); cudaFree(p.posY); //cudaFree(p.posZ);
    cudaFree(p.velX); cudaFree(p.velY); //cudaFree(p.velZ);
    cudaFree(p.lifetime); cudaFree(p.rand_state);
}



__global__ void fireKernel_SoA_FINAL(
    float* p_posX, float* p_posY,
    float* p_velX, float* p_velY,
    float* p_lifetime, unsigned int* p_rand_state,
    int numParticles,
    float dt,
    float time)
{
    // idx globale della particella
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx >= numParticles) return;

    // Dichiarazione della shared memory per un blocco di thread
    // La dimensione è specificata al lancio del kernel <<<grid, block, shmem_size>>>
    extern __shared__ float s_data[];

    // Assegnazione dei puntatori alla memoria condivisa per ogni array
    // Calcoliamo gli offset per ogni array all'interno del singolo blocco di shared memory
    float* s_posX_ptr = (float*)s_data;
    float* s_posY_ptr = (float*)(s_data + blockDim.x);
    float* s_velX_ptr = (float*)(s_data + blockDim.x * 2);
    float* s_velY_ptr = (float*)(s_data + blockDim.x * 3);
    float* s_lifetime_ptr = (float*)(s_data + blockDim.x * 4);
    unsigned int* s_rand_state_ptr = (unsigned int*)(s_data + blockDim.x * 5 * sizeof(float) / sizeof(unsigned int));

    // Caricamento dei dati dalla memoria globale alla memoria condivisa (coalesced load)
    // Ogni thread carica i suoi dati nella posizione threadIdx.x della shared memory
    s_posX_ptr[threadIdx.x] = p_posX[global_idx];
    s_posY_ptr[threadIdx.x] = p_posY[global_idx];
    s_velX_ptr[threadIdx.x] = p_velX[global_idx];
    s_velY_ptr[threadIdx.x] = p_velY[global_idx];
    s_lifetime_ptr[threadIdx.x] = p_lifetime[global_idx];
    s_rand_state_ptr[threadIdx.x] = p_rand_state[global_idx];

    // Sincronizzazione: tutti i thread nel blocco devono aver caricato i dati
    __syncthreads();

    // Ora lavoriamo con le variabili locali, inizializzate dalla shared memory
    float l_posX = s_posX_ptr[threadIdx.x];
    float l_posY = s_posY_ptr[threadIdx.x];
    float l_velX = s_velX_ptr[threadIdx.x];
    float l_velY = s_velY_ptr[threadIdx.x];
    float l_lifetime = s_lifetime_ptr[threadIdx.x];
    unsigned int local_rand_state = s_rand_state_ptr[threadIdx.x];

    const float sub_dt = dt / (float)SUB_STEPS;

    // 2. ESEGUI PIÙ CALCOLI NEI REGISTRI
    // #pragma unroll dice al compilatore di "srotolare" questo loop,
    // eliminando il costo del loop stesso e rendendolo più veloce.
#pragma unroll
    for (int i = 0; i < SUB_STEPS; ++i) {
        float current_time = time + (float)i * sub_dt;

        if (l_lifetime <= 0.0f) {
            // Posizione orizzontale di partenza leggermente casuale attorno al centro
            l_posX = (random_float(local_rand_state) - 0.5f) * 0.2f;

            l_posY = -0.9f + (random_float(local_rand_state) * 0.15f);

            // Velocità orizzontale leggermente aumentata per un effetto più "selvaggio"
            l_velX = (random_float(local_rand_state) - 0.5f) * 1.0f;

             l_velY = 4.0f + random_float(local_rand_state) * 2.0f;
            l_lifetime = 2.5f + random_float(local_rand_state) * 1.5f;
        }
        else {
            // --- BLOCCO FISICA ---
            l_velX -= l_posX * 3.0f * sub_dt; // Mantiene il fuoco centrato
            l_velY += 2.0f * sub_dt;

            float turbulence = fast_sin(fmaf(l_posY, 3.0f, fmaf(time, 2.0f, l_posX * 2.0f)))
                + fast_cos(fmaf(l_posY, 5.0f, time * 2.5f));

            l_velX = fmaf(turbulence * 0.4f, sub_dt, l_velX);

            float swirl = 0.3f * fast_sin(fmaf(time, 2.0f, l_posY * 4.0f));
            l_velX = fmaf(swirl, sub_dt, l_velX);

            l_velX *= 0.985f;
            l_velY *= 0.992f;

            l_posX = fmaf(l_velX, sub_dt, l_posX);
            l_posY = fmaf(l_velY, sub_dt, l_posY);
            l_lifetime -= sub_dt;
        }
    }

    // Sincronizzazione prima di scrivere i risultati finali
    __syncthreads();

    // Scrittura finale dalla memoria locale (registri) alla shared memory
    s_posX_ptr[threadIdx.x] = l_posX;
    s_posY_ptr[threadIdx.x] = l_posY;
    s_velX_ptr[threadIdx.x] = l_velX;
    s_velY_ptr[threadIdx.x] = l_velY;
    s_lifetime_ptr[threadIdx.x] = l_lifetime;
    s_rand_state_ptr[threadIdx.x] = local_rand_state;

    // Sincronizzazione prima di scrivere dalla shared memory alla memoria globale
    __syncthreads();

    // Scrittura finale in memoria globale (coalesced store)
    p_posX[global_idx] = s_posX_ptr[threadIdx.x];
    p_posY[global_idx] = s_posY_ptr[threadIdx.x];
    p_velX[global_idx] = s_velX_ptr[threadIdx.x];
    p_velY[global_idx] = s_velY_ptr[threadIdx.x];
    p_lifetime[global_idx] = s_lifetime_ptr[threadIdx.x];
    p_rand_state[global_idx] = s_rand_state_ptr[threadIdx.x];
}

__global__ void updateVBOKernel_SoA(
    float* p_posX, float* p_posY,
    float* p_lifetime,
    ParticleVertex* vbo_ptr, int numParticles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;
    vbo_ptr[idx].x = p_posX[idx];
    vbo_ptr[idx].y = p_posY[idx];
    vbo_ptr[idx].lifetime = p_lifetime[idx];
}

// =================================================================================
// SEZIONE 3: SHADER E MAIN
// =================================================================================
static const char* kVertexShader = R"(#version 330 core
    layout (location = 0) in vec4 aPosLifetime;
    uniform mat4 projection;
    out float vLifetime;
    void main(){
        gl_Position = projection * vec4(aPosLifetime.xyz, 1.0);
        vLifetime = aPosLifetime.w;
        float sizeFactor = clamp(vLifetime / 2.5, 0.0, 1.0);
        gl_PointSize = 1.0 + sizeFactor * 10.0; 
    }
)";
static const char* kFragmentShader = R"(#version 330 core
    in float vLifetime;
    out vec4 FragColor;
    void main(){
        float dist = distance(gl_PointCoord, vec2(0.5));
        
        // Questo crea una sfumatura molto più morbida e graduale.
        // L'esponente più alto (es. 4.0) concentra l'opacità al centro
        // e rende i bordi molto più trasparenti.
        float radialAlpha = 1.0 - pow(dist * 2.0, 4.0);
        if (radialAlpha < 0.0) discard; // Scarta i frammenti completamente trasparenti

        float t = clamp(vLifetime / 3.0, 0.0, 1.0); // Normalizza la lifetime tra 0 e 1, basandosi su una vita massima di 3 secondi

        vec3 coolColor = vec3(0.5, 0.1, 0.0);   // Rosso scuro/marrone bruciato alla base
        vec3 midColor  = vec3(0.9, 0.35, 0.0);  // Arancione caldo e vibrante
        vec3 hotColor  = vec3(1.0, 0.6, 0.1);   // Giallo-arancione, il cuore della fiamma
        vec3 coreColor = vec3(1.0, 0.9, 0.7);   // Giallo molto tenue, quasi crema (punta della fiamma)


        vec3 color;
        // Nuove soglie e transizioni per una gradazione più naturale e delicata della candela
        if (t > 0.9) { // Punta della fiamma: Giallo-arancione a Giallo tenue (ultimo 10% di vita)
            color = mix(hotColor, coreColor, (t - 0.9) / 0.1); 
        } else if (t > 0.6) { // Corpo della fiamma: Arancione caldo a Giallo-arancione (dal 60% al 90% di vita)
            color = mix(midColor, hotColor, (t - 0.6) / 0.3); 
        } else { // Base della fiamma: Rosso scuro/marrone a Arancione caldo (dal 0% al 60% di vita)
            color = mix(coolColor, midColor, t / 0.6); 
        }

        // FinalAlpha leggermente più aggressivo per una maggiore visibilità
        float finalAlpha = radialAlpha * pow(t, 0.8) * 0.4;
        
        FragColor = vec4(color, finalAlpha);
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
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(ParticleVertex), (void*)0);
    glEnableVertexAttribArray(0);

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
    h_soa.lifetime = (float*)malloc(array_size); h_soa.rand_state = (unsigned int*)malloc(rand_array_size);
    // Allocazione Device
    cudaMalloc(&d_soa.posX, array_size); cudaMalloc(&d_soa.posY, array_size);
    cudaMalloc(&d_soa.velX, array_size); cudaMalloc(&d_soa.velY, array_size);
    cudaMalloc(&d_soa.lifetime, array_size); cudaMalloc(&d_soa.rand_state, rand_array_size);

    initializeParticles_SoA(h_soa, NUM_PARTICLES);

    cudaMemcpy(d_soa.posX, h_soa.posX, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_soa.posY, h_soa.posY, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_soa.velX, h_soa.velX, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_soa.velY, h_soa.velY, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_soa.lifetime, h_soa.lifetime, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_soa.rand_state, h_soa.rand_state, rand_array_size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    cleanup_SoA_Host(h_soa); // Libera la memoria CPU, non serve più

    dim3 blockDim(512); // Dimensioni del blocco standard
    dim3 gridDim((NUM_PARTICLES + blockDim.x - 1) / blockDim.x);

    // Calcolo della dimensione della shared memory
    // 5 float arrays + 1 unsigned int array.
    // Dobbiamo fare attenzione all'allineamento. Per semplicità, allochiamo tutto come float
    // e poi castiamo per l'array di unsigned int, assicurandoci che ci sia abbastanza spazio.
    // La dimensione è in byte.
    size_t shmem_size = blockDim.x * (5 * sizeof(float) + sizeof(unsigned int));

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
        fireKernel_SoA_FINAL << <gridDim, blockDim, shmem_size >> > (
            d_soa.posX, d_soa.posY,
            d_soa.velX, d_soa.velY,
            d_soa.lifetime, d_soa.rand_state,
            NUM_PARTICLES, dt, currentTime
            );

        // Questo kernel non necessita di shared memory perché legge dalla global memory
        // e scrive direttamente nel VBO mappato, che ha un pattern di accesso coalesced.
        updateVBOKernel_SoA << <gridDim, blockDim >> > (
            d_soa.posX, d_soa.posY,
            d_soa.lifetime,
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
