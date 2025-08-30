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

struct ParticlesSoA {
    float* posX; float* posY;
    float* velX; float* velY;
    float* lifetime;
    unsigned int* rand_state;
};

struct ParticleVertex {
    float x, y, z, lifetime;
};

void initializeParticles_SoA(ParticlesSoA& p, int numParticles) {
    srand((unsigned)time(NULL));
    for (int i = 0; i < numParticles; ++i) {
        p.posX[i] = 0.0f;
        p.posY[i] =0.0f;
        p.velX[i] = 0.0f;
        p.velY[i] = 0.0f;
        p.lifetime[i] = (rand() / (float)RAND_MAX) * 2.0f + 0.5f;
        p.rand_state[i] = rand() + 1u;
    }
}

void cleanup_SoA_Host(ParticlesSoA& p) {
    free(p.posX); free(p.posY);
    free(p.velX); free(p.velY);
    free(p.lifetime); free(p.rand_state);
}

void cleanup_SoA_Device(ParticlesSoA& p) {
    cudaFree(p.posX); cudaFree(p.posY);
    cudaFree(p.velX); cudaFree(p.velY);
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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    float l_posX = p_posX[idx];
    float l_posY = p_posY[idx];
    float l_velX = p_velX[idx];
    float l_velY = p_velY[idx];
    float l_lifetime = p_lifetime[idx];
    unsigned int local_rand_state = p_rand_state[idx];
    const float sub_dt = dt / (float)SUB_STEPS;

#pragma unroll
    for (int i = 0; i < SUB_STEPS; ++i) {
        float current_time = time + (float)i * sub_dt;

        if (l_lifetime <= 0.0f) {
            // La particella era morta
            l_posX = (random_float(local_rand_state) - 0.5f) * 0.3f;
            l_posY = -0.9f + (random_float(local_rand_state) * 0.15f);

            l_velX = (random_float(local_rand_state) - 0.5f) * 1.0f;
            l_velY = 3.0f + random_float(local_rand_state) * 2.0f;

            l_lifetime = 1.0f + random_float(local_rand_state) * 1.5f;
        }
        else {
            // La particella era già viva
            l_velX = fmaf(-l_posX * 4.0f, sub_dt, l_velX);
            l_velY = fmaf(4.0f, sub_dt, l_velY);
            float turbulence = fast_sin(l_posY * 3.0f + time * 1.5f + l_posX * 2.0f)
                + fast_cos(l_posY * 5.0f + time * 2.5f);
            l_velX += turbulence * 0.4f * sub_dt;
            float swirl = 0.3f * fast_sin(time * 2.0f + l_posY * 4.0f);
            l_velX += swirl * sub_dt;
            l_velX *= 0.985f;
            l_velY *= 0.992f;
            l_posX = fmaf(l_velX, sub_dt, l_posX);
            l_posY = fmaf(l_velY, sub_dt, l_posY);
            l_lifetime -= sub_dt;
        }
    }

    p_posX[idx] = l_posX;
    p_posY[idx] = l_posY;
    p_velX[idx] = l_velX;
    p_velY[idx] = l_velY;
    p_lifetime[idx] = l_lifetime;
    p_rand_state[idx] = local_rand_state;
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

static const char* kVertexShader = R"(#version 330 core
    layout (location = 0) in vec4 aPosLifetime;
    uniform mat4 projection;
    out float vLifetime;
    void main(){
        gl_Position = projection * vec4(aPosLifetime.xyz, 1.0);
        vLifetime = aPosLifetime.w;
        float sizeFactor = clamp(vLifetime / 2.5, 0.0, 1.0);
        gl_PointSize = 1.0 + sizeFactor * 8.0; 
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

        float t = clamp(vLifetime / 3.0, 0.0, 1.0);

        vec3 coreColor = vec3(1.0, 0.8, 0.7); // Bianco incandescente
        vec3 midColor  = vec3(1.0, 0.5, 0.3);   // Giallo
        vec3 coolColor = vec3(0.9, 0.2, 0.0);   // Arancione/Rosso

        vec3 color;
        if (t > 0.2) {
            color = mix(midColor, coreColor, (t - 0.5) * 2.0);
        } else {
            color = mix(coolColor, midColor, t * 2.0);
        }

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

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(ParticleVertex), (void*)0);
    glEnableVertexAttribArray(0);

    glEnable(GL_PROGRAM_POINT_SIZE);

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
    
    cleanup_SoA_Host(h_soa);

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
            d_soa.velX, d_soa.velY, 
            d_soa.lifetime, d_soa.rand_state,
            NUM_PARTICLES, dt, currentTime
            );
        updateVBOKernel_SoA << <gridDim, blockDim >> > (
            d_soa.posX, d_soa.posY, 
            d_soa.lifetime,
            d_vbo_ptr, NUM_PARTICLES
            );



        cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);

        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);

        glUseProgram(shaderProgram);
        glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, &projection[0][0]);

        glBindVertexArray(VAO);
        glDrawArrays(GL_POINTS, 0, NUM_PARTICLES);
        glBindVertexArray(0);

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
