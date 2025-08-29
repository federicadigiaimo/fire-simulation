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
    float* posX; float* posY;
    float* velX; float* velY;
    float* lifetime;
    float* turbulence_flag;  // NUOVO
    unsigned int* rand_state;
};

// Modifica alla struttura ParticleVertex per includere il flag turbolenza
struct ParticleVertex {
    float x, y, z, lifetime;
    float turbulence_flag; // NUOVO: 1.0 se ha turbolenza attiva, 0.0 altrimenti
};

void initializeParticles_SoA(ParticlesSoA& p, int numParticles) {
    srand((unsigned)time(NULL));
    for (int i = 0; i < numParticles; ++i) {
        p.posX[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.15f;
        p.posY[i] = -0.8f + (rand() / (float)RAND_MAX) * 0.1f;
        p.velX[i] = 0.0f;
        p.velY[i] = 0.0f;
        p.lifetime[i] = (rand() / (float)RAND_MAX) * 2.0f + 0.5f;
        p.turbulence_flag[i] = 0.0f;  // NUOVO
        p.rand_state[i] = rand() + 1u;
    }
}


void cleanup_SoA_Host(ParticlesSoA& p) {
    free(p.posX); free(p.posY);
    free(p.velX); free(p.velY);
    free(p.lifetime); free(p.turbulence_flag); // NUOVO
    free(p.rand_state);
}

void cleanup_SoA_Device(ParticlesSoA& p) {
    cudaFree(p.posX); cudaFree(p.posY);
    cudaFree(p.velX); cudaFree(p.velY);
    cudaFree(p.lifetime); cudaFree(p.turbulence_flag); // NUOVO
    cudaFree(p.rand_state);
}



__global__ void fireKernel_SOA_FINAL(
    float* p_posX, float* p_posY,
    float* p_velX, float* p_velY,
    float* p_lifetime, unsigned int* p_rand_state,
    float* p_turbulence_flag,
    int numParticles,
    float dt,
    float time)
{
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx >= numParticles) return;

    // SHARED MEMORY: solo 2 float per la forza della turbolenza, che funge da "flag di blocco"
    extern __shared__ float s_data[];
    volatile float* s_turb_force_X = s_data;
    volatile float* s_turb_force_Y = s_data + 1;

    // SOLO IL THREAD 0 di ogni blocco gestisce la logica di innesco
    if (threadIdx.x == 0) {
        s_turb_force_X[0] = 0.0f;
        s_turb_force_Y[0] = 0.0f;

        // Condizione temporale per l'innesco
        float turbulence_period = 2.0f;
        float phase = fmod(time, turbulence_period);
        if (phase < dt) {
            unsigned int targetBlock = ((unsigned int)(time * 100.0f)) % gridDim.x;

            // 2. Solo il thread 0 del blocco bersaglio genera la turbolenza.
            if (blockIdx.x == targetBlock) {
                float intensity = 8.0f;
                float angle = time;
                s_turb_force_X[0] = intensity * fast_cos(angle);
                s_turb_force_Y[0] = intensity * fast_sin(angle);
            }
        }
    }
    __syncthreads();

    // Carica lo stato della particella, incluso il suo flag personale
    float l_posX = p_posX[global_idx];
    float l_posY = p_posY[global_idx];
    float l_velX = p_velX[global_idx];
    float l_velY = p_velY[global_idx];
    float l_lifetime = p_lifetime[global_idx];
    unsigned int local_rand_state = p_rand_state[global_idx];
    float my_turbulence_flag = p_turbulence_flag[global_idx];

    // Controlla se il proprio blocco ha ricevuto l'innesco
    bool block_has_turbulence = (s_turb_force_X[0] != 0.0f || s_turb_force_Y[0] != 0.0f);

    if (block_has_turbulence) {
        my_turbulence_flag = 1.0f;
    }
    // 2. ALTRIMENTI, applica il decadimento naturale.
   /* else {
        my_turbulence_flag *= 0.97f;
        if (my_turbulence_flag < 0.01f) {
            my_turbulence_flag = 0.0f;
        }
    }*/

    const float sub_dt = dt / (float)SUB_STEPS;

#pragma unroll
    for (int i = 0; i < SUB_STEPS; ++i) {
        if (l_lifetime <= 0.0f) {
            // Reset particella
            l_posX = (random_float(local_rand_state) - 0.5f) * 0.2f;
            l_posY = -0.9f + (random_float(local_rand_state) * 0.15f);
            l_velX = (random_float(local_rand_state) - 0.5f) * 1.0f;
            l_velY = 4.0f + random_float(local_rand_state) * 2.0f;
            l_lifetime = 2.5f + random_float(local_rand_state) * 1.5f;

            // CORREZIONE CRUCIALE: Resetta il flag privato quando la particella rinasce
            my_turbulence_flag = 0.0f;
        }
        else {
            // Fisica normale
            l_velX -= l_posX * 3.0f * sub_dt;
            l_velY += 2.0f * sub_dt;

            float turbulence = fast_sin(fmaf(l_posY, 3.0f, fmaf(time, 2.0f, l_posX * 2.0f)))
                + fast_cos(fmaf(l_posY, 5.0f, time * 2.5f));
            l_velX = fmaf(turbulence * 0.4f, sub_dt, l_velX);

            float swirl = 0.3f * fast_sin(fmaf(time, 2.0f, l_posY * 4.0f));
            l_velX = fmaf(swirl, sub_dt, l_velX);

            // Applica l'impulso di forza SOLO se il flag privato è attivo
            if (my_turbulence_flag > 0.0f) {
                
                float puffDirX = s_turb_force_X[0] * 0.1f; // Componente laterale debole
                float puffDirY = 0.5f;                     // Componente verticale forte

                
                float strength = 12.0f * my_turbulence_flag;

                // 3. Applica la forza
                l_velX += puffDirX * strength * sub_dt;
                l_velY += puffDirY * strength * sub_dt;

            }

            l_velX *= 0.985f;
            l_velY *= 0.992f;

            l_posX = fmaf(l_velX, sub_dt, l_posX);
            l_posY = fmaf(l_velY, sub_dt, l_posY);
            l_lifetime -= sub_dt;
        }
    }

    // Scrittura finale in memoria globale
    p_posX[global_idx] = l_posX;
    p_posY[global_idx] = l_posY;
    p_velX[global_idx] = l_velX;
    p_velY[global_idx] = l_velY;
    p_lifetime[global_idx] = l_lifetime;
    p_rand_state[global_idx] = local_rand_state;
    p_turbulence_flag[global_idx] = my_turbulence_flag;
}
// Kernel per aggiornare il VBO (invariato)
__global__ void updateVBOKernel_SoA(
    float* p_posX, float* p_posY,
    float* p_lifetime, float* p_turbulence_flag,
    ParticleVertex* vbo_ptr, int numParticles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    vbo_ptr[idx].x = p_posX[idx];
    vbo_ptr[idx].y = p_posY[idx];
    vbo_ptr[idx].z = 0.0f; // Non usato
    vbo_ptr[idx].lifetime = p_lifetime[idx];
    vbo_ptr[idx].turbulence_flag = p_turbulence_flag[idx];
}



// =================================================================================
// SEZIONE 3: SHADER E MAIN
// =================================================================================
// Vertex Shader aggiornato
static const char* kVertexShader = R"(#version 330 core
    layout (location = 0) in vec4 aPosLifetime;
    layout (location = 1) in float aTurbulenceFlag;
    uniform mat4 projection;
    out float vLifetime;
    out float vTurbulenceFlag;
    void main(){
        gl_Position = projection * vec4(aPosLifetime.xyz, 1.0);
        vLifetime = aPosLifetime.w;
        vTurbulenceFlag = aTurbulenceFlag;
        
        float sizeFactor = clamp(vLifetime / 2.5, 0.0, 1.0);
        float baseSize = 1.0 + sizeFactor * 10.0;
        
        // NUOVO: Particelle con turbolenza sono MOLTO più grandi e visibili
        if (aTurbulenceFlag > 0.1) {
            baseSize *= 2.0;  // 2 volte più grandi!
        }
        
        gl_PointSize = baseSize;
    }
)";

// Fragment Shader aggiornato
static const char* kFragmentShader = R"(#version 330 core
    in float vLifetime;
    in float vTurbulenceFlag;
    out vec4 FragColor;
    void main(){
        float dist = distance(gl_PointCoord, vec2(0.5));
        
        float radialAlpha = 1.0 - pow(dist * 2.0, 4.0);
        if (radialAlpha < 0.0) discard;

        float t = clamp(vLifetime / 3.0, 0.0, 1.0);

        // Colori normali del fuoco
        vec3 coolColor = vec3(0.5, 0.1, 0.0);
        vec3 midColor  = vec3(0.9, 0.35, 0.0);
        vec3 hotColor  = vec3(1.0, 0.6, 0.1);
        vec3 coreColor = vec3(1.0, 0.9, 0.7);

        vec3 fireColor;
        if (t > 0.9) {
            fireColor = mix(hotColor, coreColor, (t - 0.9) / 0.1); 
        } else if (t > 0.6) {
            fireColor = mix(midColor, hotColor, (t - 0.6) / 0.3); 
        } else {
            fireColor = mix(coolColor, midColor, t / 0.6); 
        }

        vec3 finalColor;
float finalAlpha = radialAlpha * pow(t, 0.8) * 0.4;

if (vTurbulenceFlag > 0.1) {
    // 1. Definisci un colore grigio cenere
    vec3 ashColor = vec3(0.6, 0.6, 0.6); 

    // 2. Mescola il colore del fuoco con il grigio.
    //    Un valore alto (0.85) lo renderà prevalentemente grigio.
    finalColor = vec3(0.6, 0.6, 0.6); 
    
    // 3. Aumenta l'opacità per renderle ben visibili
    finalAlpha *= 2.5;
} else {
    finalColor = fireColor;
}

FragColor = vec4(finalColor, finalAlpha);
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

    // Correggi anche il setup del VAO che era ridondante:
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices_size, NULL, GL_DYNAMIC_DRAW);

    // Attributo 0: aPosLifetime (vec4: x, y, z, lifetime)
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(ParticleVertex), (void*)offsetof(ParticleVertex, x));

    // Attributo 1: aTurbulenceFlag (float)  
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(ParticleVertex), (void*)offsetof(ParticleVertex, turbulence_flag));

    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glBindVertexArray(0);

    
    struct cudaGraphicsResource* cuda_vbo_resource;
    cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, VBO, cudaGraphicsRegisterFlagsNone);

    setupSinLut();
    cudaDeviceSynchronize();

    ParticlesSoA h_soa, d_soa;
    // Allocazione Host
    h_soa.posX = (float*)malloc(array_size);
    h_soa.posY = (float*)malloc(array_size);
    h_soa.velX = (float*)malloc(array_size);
    h_soa.velY = (float*)malloc(array_size);
    h_soa.lifetime = (float*)malloc(array_size);
    h_soa.turbulence_flag = (float*)malloc(array_size); // NUOVO
    h_soa.rand_state = (unsigned int*)malloc(rand_array_size);
    // Allocazione Device
    cudaMalloc(&d_soa.posX, array_size);
    cudaMalloc(&d_soa.posY, array_size);
    cudaMalloc(&d_soa.velX, array_size);
    cudaMalloc(&d_soa.velY, array_size);
    cudaMalloc(&d_soa.lifetime, array_size);
    cudaMalloc(&d_soa.turbulence_flag, array_size); // NUOVO
    cudaMalloc(&d_soa.rand_state, rand_array_size);
    initializeParticles_SoA(h_soa, NUM_PARTICLES);

    cudaMemcpy(d_soa.posX, h_soa.posX, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_soa.posY, h_soa.posY, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_soa.velX, h_soa.velX, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_soa.velY, h_soa.velY, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_soa.lifetime, h_soa.lifetime, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_soa.turbulence_flag, h_soa.turbulence_flag, array_size, cudaMemcpyHostToDevice); // NUOVO
    cudaMemcpy(d_soa.rand_state, h_soa.rand_state, rand_array_size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    cleanup_SoA_Host(h_soa); // Libera la memoria CPU, non serve più

    //unsigned char* d_turbulent;
    //cudaMalloc(&d_turbulent, NUM_PARTICLES * sizeof(unsigned char));
    //cudaMemset(d_turbulent, 0, NUM_PARTICLES * sizeof(unsigned char)); // tutti inizialmente 0

    dim3 blockDim(512); // Dimensioni del blocco standard
    dim3 gridDim((NUM_PARTICLES + blockDim.x - 1) / blockDim.x);

    // Calcolo della dimensione della shared memory
    // 5 float arrays + 1 unsigned int array.
    // Dobbiamo fare attenzione all'allineamento. Per semplicità, allochiamo tutto come float
    // e poi castiamo per l'array di unsigned int, assicurandoci che ci sia abbastanza spazio.
    // La dimensione è in byte.
    size_t shmem_size = 2 * sizeof(float);
    float lastTime = 0.0f;


    while (!glfwWindowShouldClose(window)) {
        float currentTime = (float)glfwGetTime();
        float dt = 0.016f; // Un deltaTime fisso per stabilità

        glfwPollEvents();

        ParticleVertex* d_vbo_ptr = nullptr;
        cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
        size_t num_bytes;
        cudaGraphicsResourceGetMappedPointer((void**)&d_vbo_ptr, &num_bytes, cuda_vbo_resource);

        // E nel while loop, cambia il lancio del kernel:
        fireKernel_SOA_FINAL << <gridDim, blockDim, shmem_size >> > (
            d_soa.posX, d_soa.posY,
            d_soa.velX, d_soa.velY,
            d_soa.lifetime, d_soa.rand_state,
            d_soa.turbulence_flag,  // NUOVO
            NUM_PARTICLES, dt, currentTime
            );

        updateVBOKernel_SoA << <gridDim, blockDim >> > (
            d_soa.posX, d_soa.posY,
            d_soa.lifetime, d_soa.turbulence_flag,  // NUOVO
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
    //cudaFree(d_turbulent);

    glDeleteProgram(shaderProgram);
    glDeleteBuffers(1, &VBO);
    glDeleteVertexArrays(1, &VAO);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
