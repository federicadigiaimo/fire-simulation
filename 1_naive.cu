#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <curand_kernel.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

//#define SEQUENTIAL_BASELINE
#define GPU_NAIVE_PARALLEL

// Stato particella
typedef struct {
    float posX, posY;
    float velX, velY;
    float lifetime;
} Particle;

// Dati per rendering particella
typedef struct {
    float x, y;
} ParticleVertex;

float get_random_float() {
    return (float)rand() / (float)RAND_MAX;
}

void initializeParticles(Particle* particles, int numParticles) {
    for (int i = 0; i < numParticles; i++) {
        if (i % 1000 == 0) {
            for (int j = 0; j < 3; j++) {
                rand();
            }
        }

        particles[i].lifetime = get_random_float() * 2.0f + 0.5f;

        particles[i].posX = 0.0f;
        particles[i].posY = 0.0f;

        particles[i].velX = (get_random_float() - 0.5f) * 0.1f;
        particles[i].velY = get_random_float() * 0.2f + 0.1f;
    }
}

__global__ void initCurandKernel(curandState* states, int numParticles, unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) {
        return;
    }
    curand_init(seed, idx, 0, &states[idx]);
}

#ifdef SEQUENTIAL_BASELINE
void updateParticlesCPU(Particle* particles, int numParticles, float dt, float time)
{

    for (int idx = 0; idx < numParticles; idx++) {
        Particle p = particles[idx];

        //Particella viva
        if (p.lifetime > 0.0f) {
            p.velX -= p.posX * 1.2f * dt;

            float turbulence = sin(p.posY * 3.0f + time * 1.5f + p.posX * 2.0f)
                + cos(p.posY * 5.0f + time * 2.5f);
            p.velX += turbulence * 0.4f * dt;

            float swirl = 0.3f * sin(time * 2.0f + p.posY * 4.0f);
            p.velX += swirl * dt;

            p.velY += 0.8f * dt;

            p.velX *= 0.985f;
            p.velY *= 0.992f;

            p.posX += p.velX * dt;
            p.posY += p.velY * dt;
            p.lifetime -= dt;

        }
        // Rinascita
        else {
            p.posX = (get_random_float() - 0.5f) * 0.8f;
            p.posY = -0.8f + get_random_float() * 0.05f;

            p.velX = (get_random_float() - 0.5f) * 0.8f;
            p.velY = get_random_float() * 1.2f + 1.0f;

            p.lifetime = 0.8f + get_random_float() * 1.2f;
        }

        particles[idx] = p;
    }
}
#endif

#ifdef  GPU_NAIVE_PARALLEL
__global__ void updateParticlesKernel(Particle* particles, ParticleVertex* vbo_ptr,
    int numParticles, float dt, curandState* randStates, float time)
{

    // Calcolo indirizzo globale thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    curandState localState = randStates[idx];
    Particle p = particles[idx];

    //Particella viva
    if (p.lifetime > 0.0f)
    {

        p.velX -= p.posX * 1.2f * dt;

        float turbulence = sin(p.posY * 3.0f + time * 1.5f + p.posX * 2.0f)
            + cos(p.posY * 5.0f + time * 2.5f);
        p.velX += turbulence * 0.4f * dt;

        float swirl = 0.3f * sin(time * 2.0f + p.posY * 4.0f);
        p.velX += swirl * dt;

        p.velY += 0.5f * dt;

        p.velX *= 0.985f;
        p.velY *= 0.992f;

        p.posX += p.velX * dt;
        p.posY += p.velY * dt;

        p.lifetime -= dt;

    }
    // Rinascita
    else
    {
        p.posX = (curand_uniform(&localState) - 0.5f) * 0.1f;
        p.posY = -0.8f + curand_uniform(&localState) * 0.05f;

        p.velX = (curand_uniform(&localState) - 0.5f) * 0.8f;
        p.velY = curand_uniform(&localState) * 1.2f + 1.0f;

        p.lifetime = 0.8f + curand_uniform(&localState) * 1.2f;
    }

    particles[idx] = p;
    vbo_ptr[idx].x = p.posX;
    vbo_ptr[idx].y = p.posY;

    randStates[idx] = localState;
}
#endif

static const char* kVertexShader = R"(#version 330 core
layout (location = 0) in vec2 aPos;   // solo x, y
uniform mat4 projection;

void main(){
    gl_Position = projection * vec4(aPos, 0.0, 1.0);  
    gl_PointSize = 6.0; // grandezza fissa della particella
}
)";
static const char* kFragmentShader = R"(#version 330 core
out vec4 FragColor;

void main(){
    float dist = distance(gl_PointCoord, vec2(0.5));  
    if (dist > 0.5) discard; // bordo tondo

    float alpha = 1.0 - dist * 2.0; // dissolvenza dal centro al bordo
    vec3 color = vec3(1.0, 0.6, 0.1); // arancio fisso

    FragColor = vec4(color, alpha);
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
    size_t size = NUM_PARTICLES * sizeof(Particle);
    size_t vertices_size = NUM_PARTICLES * sizeof(ParticleVertex);

    GLuint VBO, VAO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices_size, NULL, GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(ParticleVertex), (void*)0);
    glEnableVertexAttribArray(0);

    glEnable(GL_PROGRAM_POINT_SIZE);

    glBindVertexArray(0);

    struct cudaGraphicsResource* cuda_vbo_resource;
    cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, VBO, cudaGraphicsRegisterFlagsNone);
    
    cudaDeviceSynchronize();



#if defined(SEQUENTIAL_BASELINE)

    printf("Esecuzione in modalità: SEQUENTIAL_BASELINE (CPU)\n");

    // Allocazione memoria particelle
    Particle* h_particles = (Particle*)malloc(size);
    ParticleVertex* h_vertices = (ParticleVertex*)malloc(vertices_size);

    // Inizializzazione particelle
    initializeParticles(h_particles, NUM_PARTICLES);

    // Pulizia vertici
    memset(h_vertices, 0, vertices_size);

    // Misurazione del tempo
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

        for (int i = 0; i < NUM_PARTICLES; i++) {
            h_vertices[i].x = h_particles[i].posX;
            h_vertices[i].y = h_particles[i].posY;

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
    const int BLOCK_SIZE = 512;
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

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        if (glfwWindowShouldClose(window)) {
            break;
        }

        ParticleVertex* d_vbo_ptr;
        cudaError_t mapResult = cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
        if (mapResult != cudaSuccess) {
            printf("Errore mapping CUDA resource: %s\n", cudaGetErrorString(mapResult));
            break;
        }

        size_t num_bytes;
        cudaError_t ptrResult = cudaGraphicsResourceGetMappedPointer((void**)&d_vbo_ptr, &num_bytes, cuda_vbo_resource);
        if (ptrResult != cudaSuccess) {
            printf("Errore getting mapped pointer: %s\n", cudaGetErrorString(ptrResult));
            cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
            break;
        }

        float t = glfwGetTime();

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);

        updateParticlesKernel << <gridSize, BLOCK_SIZE >> > 
            (d_particles, d_vbo_ptr, NUM_PARTICLES, 0.016f, d_randStates, t);

        cudaError_t kernelResult = cudaGetLastError();
        if (kernelResult != cudaSuccess) {
            printf("Errore kernel CUDA: %s\n", cudaGetErrorString(kernelResult));
            cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
            break;
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        static int frameCount = 0;
        if (frameCount % 60 == 0) {
            printf("Tempo di esecuzione GPU per un update: %.4f ms\n", milliseconds);
        }
        frameCount++;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        // DISEGNO
        // Restituzione vbo a OpenGL per disegnare
        cudaError_t unmapResult = cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
        if (unmapResult != cudaSuccess) {
            printf("Errore unmapping CUDA resource: %s\n", cudaGetErrorString(unmapResult));
            break;
        }

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

    }

    cudaGraphicsUnregisterResource(cuda_vbo_resource);
    cudaDeviceSynchronize();

    cudaFree(d_particles);
    cudaFree(d_randStates);

    cudaDeviceReset();

    printf("Cleanup completato.\n");
#endif

    glDeleteProgram(shaderProgram);
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}