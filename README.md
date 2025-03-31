# CUDA-Based Electromagnetic Wave Simulation

This project implements a 2D Transmission Line Matrix (TLM) method to simulate electromagnetic wave propagation using CUDA for parallelisation. The simulation is based on finite-difference techniques, enabling high-performance modeling on GPU devices.

## Features
- GPU-accelerated 2D TLM solver
- Source injection and output sampling
- Boundary reflections with configurable coefficients
- Efficient memory management and synchronisation
- Time-domain simulation with customisable steps
- Outputs waveforms in a plain text file

## File Structure
- `main.cu`: Main simulation file, includes CUDA kernel implementations and simulation loop
- `outputCu.out`: Output file containing time and voltage values at a specific mesh point

## Physics Behind the Code
The code models a 2D grid where each node exchanges voltage/current information with its neighbors via TLM scattering and connection processes. The wave is injected using a Gaussian pulse and propagates across the mesh while obeying Maxwell’s equations in a discretised form.

## How It Works

1. **Initialisation**: 
   - A 2D mesh of nodes is created and initialised to zero.
   - Voltage components (`V1`, `V2`, `V3`, `V4`) are stored in flattened arrays.

2. **Time-Stepping Loop**:
   - A Gaussian pulse is injected at a specific mesh location.
   - The wave is scattered, connected to neighboring nodes, and boundary conditions are applied.
   - Output voltages are saved at a predefined observation point.

3. **Output**:
   - The time-domain waveform of the electric field at the observation point is saved in `outputCu.out`.
