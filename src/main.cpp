#include "simulation.cuh"
#include <iostream>
#include <SFML/Graphics.hpp>
#include <optional>
#include <sstream>
#include "fps.h"

int main(){


    // Initialize variables 
    FPS fps;
    unsigned int screensize = 1000;
    float particleCount = 10000;
    int simSpeed = 0;


    // Create Particles
    std::vector<Particle> particles = createParticles(particleCount,screensize);
    sf::VertexArray points(sf::PrimitiveType::Points,particleCount);


    // Create rendering window
    sf::RenderWindow window(sf::VideoMode({screensize, screensize}), "Simulation");
    window.setFramerateLimit(165);


    while (window.isOpen())
    {
        // Handle events
        while (const std::optional event = window.pollEvent())
        {

            if (event->is<sf::Event::Closed>())
                window.close();
        }


        // Compute velocities with multiple substeps for stability
        if (simSpeed == 3)
        {
            sf::Vector2i mousePos = sf::Mouse::getPosition(window);
            // Run multiple substeps per frame for numerical stability
            for(int substep = 0; substep < 20; substep++) {
                compute(particles, screensize, particleCount, mousePos);
            }
            simSpeed = 0;
        }
        simSpeed++;


        // Draw particles
        window.clear();
        for (size_t i = 0; i < particles.size(); i++)
        {
            particles[i].updatePos();
            points[i] = particles[i].circle;
        }
        window.draw(points);


        // Update fps counter
        fps.update();
        std::ostringstream ss;
        ss << fps.getFPS();
        window.setTitle(ss.str());


        // Display to window
        window.display();
        }
    std::cout << particles[particleCount/2+20].density;
    return 0;
}