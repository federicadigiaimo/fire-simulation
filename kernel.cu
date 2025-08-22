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

#define BLOCK_SIZE 256

//#define SEQUENTIAL_BASELINE
//#define GPU_NAIVE_PARALLEL
 #define SHARED_MEM_VERSION
// #define DIVERGENCE_VERSION

// Stato particella
typedef struct {
    float posX, posY, posZ;
    float velX, velY, velZ;
    float lifetime;
} Particle;

// Dati per rendering particella
typedef struct {
    float x, y, z;
    float r, g, b, a;
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

            // Spinta verso l�alto, aria calda che sale
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


            // Velocit� iniziale rinascita
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
    int numParticles, float dt, curandState* randStates, float time) 
{
    
    // Calcolo indirizzo globale thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles)
    {
        return;
    }

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

        // Spinta verso l�alto, aria calda che sale
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

        // Velocit� iniziale rinascita
        p.velX = (curand_uniform(&localState) - 0.5f) * 0.8f;
        p.velY = curand_uniform(&localState) * 1.2f + 1.0f;
        p.velZ = 0.0f;

        // Tempo di vita della particella
        p.lifetime = 0.8f + curand_uniform(&localState) * 1.2f;
    }

    particles[idx] = p;
    
    // Scrivo direttamente su VBO aggiornamento dati di posizione e colore delle particelle
    vbo_ptr[idx].x = p.posX;
    vbo_ptr[idx].y = p.posY;
    vbo_ptr[idx].z = 0.0f;


    randStates[idx] = localState;
}

#endif

#ifdef SHARED_MEM_VERSION
__global__ void updateParticlesKernel(Particle* particles, ParticleVertex* vbo_ptr,
    int numParticles, float dt, curandState* randStates, float time)
{
    // Ogni blocco avr� la sua copia privata di questo array
    __shared__ Particle shared_particles[BLOCK_SIZE];

    // Indice locale: la posizione del thread all'interno del blocco
    int local_idx = threadIdx.x;

    // Calcolo indirizzo globale thread
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx >= numParticles)
    {
        return;
    }
    else if (global_idx < numParticles)
    {
      // Ogni thread del blocco carica UNA particella dalla memoria globale
      // alla sua posizione corrispondente nell'array condiviso
        shared_particles[local_idx] = particles[global_idx];
    }

    // Sincronizzazione
   // Questa barriera assicura che NESSUN thread proceda
   // finch� TUTTI i thread del blocco non hanno completato la riga precedente.
   // Ora siamo sicuri che 'shared_particles' � completamente piena.
    __syncthreads();


    if (global_idx < numParticles) {

        // Mantieni lo stato di curand (non � condiviso).
        curandState localState = randStates[global_idx];
        // Leggi lo stato della particella DALLA SHARED MEMORY.
        // Questo accesso � ordini di grandezza pi� veloce di una lettura da memoria globale.
        Particle p = shared_particles[local_idx];


        //Particella viva
        if (p.lifetime > 0.0f)
        {
            if (local_idx > 0) { // Controlla di non essere il primo thread del blocco
                Particle neighbor_left = shared_particles[local_idx - 1];
                if (neighbor_left.velY > p.velY) {
                    p.velX += 0.05f * (neighbor_left.velY - p.velY) * dt;
                }
            }
            // Forza di convergenza, per mantenere forma della fiamma
            p.velX -= p.posX * 1.2f * dt;

            // Turbolenza
            float turbulence = sin(p.posY * 3.0f + time * 1.5f + p.posX * 2.0f)
                + cos(p.posY * 5.0f + time * 2.5f);
            p.velX += turbulence * 0.4f * dt;

            // Swirl
            float swirl = 0.3f * sin(time * 2.0f + p.posY * 4.0f);
            p.velX += swirl * dt;

            // Spinta verso l�alto, aria calda che sale
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

            // Velocit� iniziale rinascita
            p.velX = (curand_uniform(&localState) - 0.5f) * 0.8f;
            p.velY = curand_uniform(&localState) * 1.2f + 1.0f;
            p.velZ = 0.0f;

            // Tempo di vita della particella
            p.lifetime = 0.8f + curand_uniform(&localState) * 1.2f;
        }

        particles[global_idx] = p;

        // Scrivo direttamente su VBO aggiornamento dati di posizione e colore delle particelle
        vbo_ptr[global_idx].x = p.posX;
        vbo_ptr[global_idx].y = p.posY;
        vbo_ptr[global_idx].z = 0.0f;


        randStates[global_idx] = localState;
    }
}
#endif

const char* vertexShaderSource = "#version 330 core\n"
"layout (location = 0) in vec3 aPos;\n"
"uniform mat4 projection;\n"
"void main() {\n"
"   gl_Position = projection * vec4(aPos, 1.0);\n"
"   gl_PointSize = 50.0;\n"
"}\0";


const char* fragmentShaderSource = "#version 330 core\n"
"out vec4 FragColor;\n"
"void main() {\n"
"   float dist = distance(gl_PointCoord, vec2(0.5));\n"
"   float alpha = 1.0 - pow(dist * 2.0, 2.0);\n"
"   if (alpha < 0.05) discard;\n"
"\n"
"   // Altezza\n"
"   float t = gl_FragCoord.y / 800.0;\n"
"\n"
"   vec3 baseColor = vec3(1.0, 0.8, 0.6);   // bianco-giallo\n"
"   vec3 midColor  = vec3(1.0, 0.5, 0.0);   // arancione\n"
"   vec3 topColor  = vec3(1.0, 0.0, 0.0);   // rosso\n"
"\n"
"   // Interpolazione\n"
"   vec3 fireColor = mix(baseColor, midColor, t);\n"
"   fireColor = mix(fireColor, topColor, t * t);\n"
"\n"
"   // Output finale\n"
"   FragColor = vec4(fireColor, alpha);\n"
"}\n\0";


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

#elif defined(GPU_NAIVE_PARALLEL) || defined(SHARED_MEM_VERSION)
    // Definizione dimensione thread block e grid
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


