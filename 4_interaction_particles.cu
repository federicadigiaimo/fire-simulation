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
    float* turbulence_flag;
    unsigned int* rand_state;
};

struct ParticleVertex {
    float x, y, lifetime;
    float turbulence_flag;
};


void initializeParticles_SoA(ParticlesSoA& p, int numParticles) {
    srand((unsigned)time(NULL));

    for (int i = 0; i < numParticles; ++i) {
        p.posX[i] = -1000.0f;
        p.posY[i] = -1000.0f;

        p.velX[i] = 0.0f;
        p.velY[i] = 0.0f;

        p.lifetime[i] = ((float)rand() / (float)RAND_MAX) * 2.5f;

        p.turbulence_flag[i] = 0.0f;

        unsigned int seed = i * 1324853;
        p.rand_state[i] = seed;
    }
}


void cleanup_SoA_Host(ParticlesSoA& p) {
    free(p.posX); free(p.posY);
    free(p.velX); free(p.velY);
    free(p.lifetime); free(p.turbulence_flag);
    free(p.rand_state);
}

void cleanup_SoA_Device(ParticlesSoA& p) {
    cudaFree(p.posX); cudaFree(p.posY);
    cudaFree(p.velX); cudaFree(p.velY);
    cudaFree(p.lifetime); cudaFree(p.turbulence_flag);
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

    extern __shared__ float s_data[];
    volatile float* s_turb_force_X = s_data;
    volatile float* s_turb_force_Y = s_data + 1;

    if (threadIdx.x == 0) {
        s_turb_force_X[0] = 0.0f;
        s_turb_force_Y[0] = 0.0f;

        float turbulence_period = 1.8f + (fast_sin(time * 0.1f) + 1.0f) * 1.0f;
        float phase = fmod(time, turbulence_period);
        if (phase < dt) {
            unsigned int targetBlock = ((unsigned int)(time * 100.0f)) % gridDim.x;

            if (blockIdx.x == targetBlock) {
                float intensity = 8.0f;
                float angle = time;
                s_turb_force_X[0] = intensity * fast_cos(angle);
                s_turb_force_Y[0] = intensity * fast_sin(angle);
            }
        }
    }
    __syncthreads();

    float l_posX = p_posX[global_idx];
    float l_posY = p_posY[global_idx];
    float l_velX = p_velX[global_idx];
    float l_velY = p_velY[global_idx];
    float l_lifetime = p_lifetime[global_idx];
    unsigned int local_rand_state = p_rand_state[global_idx];
    float my_turbulence_flag = p_turbulence_flag[global_idx];

    float cached_turb_X = s_turb_force_X[0];
    float cached_turb_Y = s_turb_force_Y[0];
    bool block_has_turbulence = (cached_turb_X != 0.0f || cached_turb_Y != 0.0f);

    if (block_has_turbulence) {
        my_turbulence_flag = 1.0f;
    }

    const float sub_dt = dt / (float)SUB_STEPS;

#pragma unroll
    for (int i = 0; i < SUB_STEPS; ++i) {
        if (l_lifetime <= 0.0f) {
            l_posX = (random_float(local_rand_state) - 0.5f) * 0.2f;
            l_posY = -0.9f + (random_float(local_rand_state) * 0.15f);
            l_velX = (random_float(local_rand_state) - 0.5f) * 1.0f;
            l_velY = 4.0f + random_float(local_rand_state) * 2.0f;
            l_lifetime = 2.5f + random_float(local_rand_state) * 1.5f;
            my_turbulence_flag = 0.0f;
        }
        else {
            if (l_posY < -100.0f) {
                l_lifetime -= sub_dt;
                continue;
            }
            l_velX -= l_posX * 3.0f * sub_dt;
            l_velY += 2.0f * sub_dt;

            float turbulence = sinf(fmaf(l_posY, 3.0f, fmaf(time, 2.0f, l_posX * 2.0f)))
                + cosf(fmaf(l_posY, 5.0f, time * 2.5f));
            l_velX = fmaf(turbulence * 0.4f, sub_dt, l_velX);

            float swirl = 0.3f * sinf(fmaf(time, 2.0f, l_posY * 4.0f));
            l_velX = fmaf(swirl, sub_dt, l_velX);

            if (my_turbulence_flag > 0.0f)
            {
                float puffDirX = cached_turb_X * 0.1f;
                float strength = 12.0f * my_turbulence_flag;
                l_velX += puffDirX * strength * sub_dt;
                l_velY += 0.5f * strength * sub_dt * 0.5f;
            }

            l_velX *= 0.985f;
            l_velY *= 0.992f;

            l_posX = fmaf(l_velX, sub_dt, l_posX);
            l_posY = fmaf(l_velY, sub_dt, l_posY);
            l_lifetime -= sub_dt;
        }
    }
    p_posX[global_idx] = l_posX;
    p_posY[global_idx] = l_posY;
    p_velX[global_idx] = l_velX;
    p_velY[global_idx] = l_velY;
    p_lifetime[global_idx] = l_lifetime;
    p_rand_state[global_idx] = local_rand_state;
    p_turbulence_flag[global_idx] = my_turbulence_flag;
}

__global__ void updateVBOKernel_SoA(
    float* __restrict__ p_posX,
    float* __restrict__ p_posY,
    float* __restrict__ p_lifetime,
    float* __restrict__ p_turbulence_flag,
    ParticleVertex* __restrict__ vbo_ptr,
    int numParticles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    float4 packed_data;
    packed_data.x = p_posX[idx];
    packed_data.y = p_posY[idx];
    packed_data.z = p_lifetime[idx];
    packed_data.w = p_turbulence_flag[idx];

    *(float4*)&vbo_ptr[idx] = packed_data;
}

static const char* kFragmentShader = R"(#version 330 core
    in float vLifetime;
    in float vTurbulenceFlag;
    
    in vec3 worldPos;
    
    out vec4 FragColor;
    
    void main(){
        float dist = distance(gl_PointCoord, vec2(0.5));
        
        float radialAlpha = 1.0 - pow(dist * 2.0, 4.0);
        if (radialAlpha < 0.0) discard;

        float t = clamp(vLifetime / 3.0, 0.0, 1.0);

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
            vec3 smokeColor = vec3(0.8, 0.8, 0.9);
            finalColor = smokeColor;
            
            float fireTopHeight = 2.0f;
            float heightFactor = smoothstep(fireTopHeight - 0.2f, fireTopHeight + 0.3f, worldPos.y);
            finalAlpha = finalAlpha * 0.3f * heightFactor;
            
            if (finalAlpha < 0.01) discard;
            
        } else {
            finalColor = fireColor;
        }

        FragColor = vec4(finalColor, finalAlpha);
    }
)";

static const char* kVertexShader = R"(#version 330 core
    layout (location = 0) in vec4 aData;
    
    uniform mat4 projection;

    out float vLifetime;
    out float vTurbulenceFlag;
    out vec3 worldPos;
    
    void main(){
        float posX = aData.x;
        float posY = aData.y;
        float posZ = 0.0;
        
        vLifetime = aData.z;
        vTurbulenceFlag = aData.w;

        worldPos = vec3(posX, posY, posZ);
        gl_Position = projection * vec4(worldPos, 1.0);
        
        float sizeFactor = clamp(vLifetime / 2.5, 0.0, 1.0);
        float baseSize = 1.0 + sizeFactor;
        if (vTurbulenceFlag > 0.1) {
            baseSize = 1.0 + sizeFactor * 30.0;
        }
        gl_PointSize = baseSize;
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

    const int NUM_PARTICLES = 1048576;//2097152;//67108864;//16777216; //524288;///65536;
    size_t vertices_size = NUM_PARTICLES * sizeof(ParticleVertex);
    size_t array_size = NUM_PARTICLES * sizeof(float);
    size_t rand_array_size = NUM_PARTICLES * sizeof(unsigned int);

    GLuint VBO, VAO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);

    glBufferData(GL_ARRAY_BUFFER, vertices_size, NULL, GL_DYNAMIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(ParticleVertex), (void*)0);

    glDisableVertexAttribArray(1);

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
    h_soa.turbulence_flag = (float*)malloc(array_size);
    h_soa.rand_state = (unsigned int*)malloc(rand_array_size);
    // Allocazione Device
    cudaMalloc(&d_soa.posX, array_size);
    cudaMalloc(&d_soa.posY, array_size);
    cudaMalloc(&d_soa.velX, array_size);
    cudaMalloc(&d_soa.velY, array_size);
    cudaMalloc(&d_soa.lifetime, array_size);
    cudaMalloc(&d_soa.turbulence_flag, array_size);
    cudaMalloc(&d_soa.rand_state, rand_array_size);
    initializeParticles_SoA(h_soa, NUM_PARTICLES);

    cudaMemcpy(d_soa.posX, h_soa.posX, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_soa.posY, h_soa.posY, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_soa.velX, h_soa.velX, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_soa.velY, h_soa.velY, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_soa.lifetime, h_soa.lifetime, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_soa.turbulence_flag, h_soa.turbulence_flag, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_soa.rand_state, h_soa.rand_state, rand_array_size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    cleanup_SoA_Host(h_soa);

    dim3 blockDim(512);
    dim3 gridDim((NUM_PARTICLES + blockDim.x - 1) / blockDim.x);

    size_t shmem_size = 2 * sizeof(float);
    float lastTime = 0.0f;


    while (!glfwWindowShouldClose(window)) {
        float currentTime = (float)glfwGetTime();
        float dt = 0.016f;

        glfwPollEvents();

        ParticleVertex* d_vbo_ptr = nullptr;
        cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
        size_t num_bytes;
        cudaGraphicsResourceGetMappedPointer((void**)&d_vbo_ptr, &num_bytes, cuda_vbo_resource);


        fireKernel_SOA_FINAL << <gridDim, blockDim, shmem_size >> > (
            d_soa.posX, d_soa.posY,
            d_soa.velX, d_soa.velY,
            d_soa.lifetime, d_soa.rand_state,
            d_soa.turbulence_flag,
            NUM_PARTICLES, dt, currentTime
            );

        updateVBOKernel_SoA << <gridDim, blockDim >> > (
            d_soa.posX, d_soa.posY,
            d_soa.lifetime, d_soa.turbulence_flag,
            d_vbo_ptr, NUM_PARTICLES
            );



        cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
        cudaDeviceSynchronize();

        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

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