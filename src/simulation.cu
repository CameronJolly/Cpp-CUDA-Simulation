#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <iostream>
#include <chrono>
#include <SFML/Graphics.hpp>
#include "particle.h"
#include <algorithm>
#include <math_constants.h>

using namespace std;

// SPH Constants
__constant__ float h = 20.0f;           // Smoothing radius
__constant__ float h2 = 400.0f;         // h^2
__constant__ float mass = 1.0f;         // Particle mass
__constant__ float restDensity = 1.0f;  // Rest density
__constant__ float gasConstant = 100.0f; // Gas constant for pressure
__constant__ float dt = 0.011f;         // Time step (much smaller for stability)

__global__ void resetDensity(Particle* particles, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;
    particles[index].density = 0;
}

// Compute density using Poly6 kernel
__global__ void densityKernel(Particle* particles, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    float density = 0.0f;
    float poly6Coeff = 4.0f / (CUDART_PI_F * powf(h, 8.0f));

    for (int j = 0; j < size; j++) {
        if (i == j) continue;

        float dx = particles[i].pos.x - particles[j].pos.x;
        float dy = particles[i].pos.y - particles[j].pos.y;
        float r2 = dx * dx + dy * dy;

        if (r2 < h2) {
            float diff = h2 - r2;
            density += mass * poly6Coeff * diff * diff * diff;
        }
    }
    
    // Add self-contribution
    particles[i].density = density + mass * poly6Coeff * powf(h2, 3.0f);
}

// Compute pressure from density using equation of state
__global__ void pressureKernel(Particle* particles, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;
    
    // Tait equation of state: p = k * (ρ - ρ₀)
    particles[index].pressure = gasConstant * (particles[index].density - restDensity);
}

// Compute pressure forces using Spiky kernel gradient
__global__ void pressureForceKernel(Particle* particles, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    float fx = 0.0f, fy = 0.0f;
    float spikyCoeff = -30.0f / (CUDART_PI_F * powf(h, 5.0f));

    for (int j = 0; j < size; j++) {
        if (i == j) continue;

        float dx = particles[i].pos.x - particles[j].pos.x;
        float dy = particles[i].pos.y - particles[j].pos.y;
        float r2 = dx * dx + dy * dy;

        if (r2 < h2 && r2 > 1e-6f) {
            float r = sqrtf(r2);
            float diff = h - r;
            
            float pressureTerm = particles[i].pressure / (particles[i].density * particles[i].density) +
                               particles[j].pressure / (particles[j].density * particles[j].density);
            float forceMagnitude = -mass * mass * pressureTerm * spikyCoeff * diff * diff / r;
            
            fx += forceMagnitude * dx;
            fy += forceMagnitude * dy;
        }
    }

    particles[i].vel.x += fx * dt;
    particles[i].vel.y += fy * dt;
}

// Kernel to compute local physics like wall collisions or gravity
__global__ void generalKenel(Particle* particles, int size, int threadPerBlock, int screenSize, int mouseX, int mouseY)
{
            float gravity = 10.0;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < size){
        // Apply Gravity
        particles[index].vel.y += gravity * dt;
        
        // Update position based on velocity
        particles[index].pos.x += particles[index].vel.x * dt;   
        particles[index].pos.y += particles[index].vel.y * dt; 
        
        // Keep particles in bounds with damping
        float damping = 0.8f;
        if (particles[index].pos.x <= 0){
            particles[index].pos.x = 0;
            particles[index].vel.x *= -damping;
        } else if (particles[index].pos.x >= screenSize) {
            particles[index].pos.x = screenSize;
            particles[index].vel.x *= -damping;
        }
        if (particles[index].pos.y <= 0) {
            particles[index].pos.y = 0;
            particles[index].vel.y *= -damping;
        } else if (particles[index].pos.y >= screenSize-1) {
            particles[index].pos.y = screenSize-1;
            particles[index].vel.y *= -damping;
        }
    }  
}

void compute(vector<Particle>& particles, int screenSize, int totalParticles, sf::Vector2i mousePos){

    int size = particles.size();

    Particle* dParticles;
    cudaMalloc(&dParticles, size*sizeof(Particle));
    cudaMemcpy(dParticles,particles.data(),size*sizeof(Particle),cudaMemcpyHostToDevice);
    
    int threadPerBlock = 256;
    int numBlocks = (totalParticles + threadPerBlock - 1) / threadPerBlock;

    // SPH computation pipeline
    resetDensity<<<numBlocks, threadPerBlock>>>(dParticles, size);
    cudaDeviceSynchronize();

    densityKernel<<<numBlocks, threadPerBlock>>>(dParticles, size);
    cudaDeviceSynchronize();

    pressureKernel<<<numBlocks, threadPerBlock>>>(dParticles, size);
    cudaDeviceSynchronize();

    pressureForceKernel<<<numBlocks, threadPerBlock>>>(dParticles, size);
    cudaDeviceSynchronize();

    generalKenel<<<numBlocks, threadPerBlock>>>(dParticles, size, threadPerBlock, screenSize, mousePos.x, mousePos.y);
    cudaDeviceSynchronize();

    cudaMemcpy(particles.data(),dParticles,size*sizeof(Particle),cudaMemcpyDeviceToHost);
    cudaFree(dParticles);

    // Color particles based on pressure (red = low pressure, yellow = high pressure)
    for (size_t i = 0; i < particles.size(); i++)
    {
        float normalizedDensity = min(particles[i].density*4000.0f, 255.0f);
        unsigned char colorValue = static_cast<unsigned char>(normalizedDensity);

        // Set color: Red base, green intensity changes with density
        particles[i].circle.color.r = 255;
        particles[i].circle.color.g = colorValue;
        particles[i].circle.color.b = 0;
    }
}

vector<Particle> createParticles(int totalParticles, int screenSize){
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> posDistrib(screenSize*0.2f, screenSize*0.8f); 
    uniform_real_distribution<float> velDistrib(-10.f, 10.f); 

    vector<Particle> particles;
    particles.reserve(totalParticles);

    for (int i = 0; i < totalParticles; i++) {
        float xpos = posDistrib(gen);
        float ypos = posDistrib(gen);
        float xvel = velDistrib(gen);
        float yvel = velDistrib(gen);
        particles.emplace_back(Particle({xpos, ypos}, {xvel, yvel}));
    }

    return particles;
}