# Simulation of Nonequilibrium Molecular Dynamics with Numba and CuPy

### Author: Marco Cavazza

### Academic Year: 2021-2022

### University: University of Modena and Reggio Emilia

### Department: Department of Physics, Physical, Computer and Mathematical Sciences

---

## Project Overview

This project explores the use of GPUs (Graphics Processing Units) to accelerate nonequilibrium molecular dynamics simulations. The study leverages Python programming, specifically using Numba and CuPy libraries, to enhance performance through parallel computation.

### Key Sections

1. **GPU Programming**
   - **GPU Architecture**: Discussion on the fundamental differences between CPU and GPU architectures, emphasizing the parallel processing capabilities of GPUs.
   - **GPGPU (General-Purpose Computing on GPUs)**: Overview of the evolution of GPUs from purely graphical processing to general-purpose computation, allowing for the efficient execution of highly parallel tasks.
   - **Python Acceleration Techniques**: Introduction to Python libraries (NumPy, CuPy, Numba) that facilitate accelerated computation by utilizing GPU resources.

2. **Molecular Dynamics**
   - **Classical Mechanics Models**: Examination of classical models used in molecular dynamics, including potential functions like the Lennard-Jones potential.
   - **Numerical Integration**: Detailed explanation of numerical methods to integrate equations of motion, with a focus on the velocity-Verlet algorithm.

3. **Numerical Simulation**
   - **Code Implementation**: Description of the molecular dynamics simulation code, including the use of linked-cell algorithms for efficient force calculations.
   - **Parallelization Approaches**: Exploration of different parallelization strategies using Numba and CUDA to optimize performance.
   - **Optimization with NumPy and CuPy**: Techniques to further enhance simulation efficiency by leveraging optimized libraries for GPU computation.

### Results

The project demonstrates significant performance improvements by utilizing GPU acceleration for molecular dynamics simulations. Comparative analyses between CPU-based and GPU-based implementations highlight the efficiency gains achieved through parallel computation.

### Conclusion

The study successfully showcases the potential of GPU acceleration in scientific computing, particularly for computationally intensive tasks like nonequilibrium molecular dynamics simulations. The use of Numba and CuPy in Python provides a powerful and flexible approach to harnessing the computational power of GPUs, offering substantial speedups over traditional CPU-based methods.

---

## Files Included

- **main.pdf**: The main document detailing the entire project, including theoretical background, methodology, implementation, and results.
- **nvtg2_sample_two_index.ipynb**: Jupyter notebook containing code samples and simulations related to the project.
- **run_total_CP.ipynb**: Jupyter notebook with code for specific simulation runs under constant pressure conditions.
- **run_total_NP.ipynb**: Jupyter notebook with code for specific simulation runs under non-equilibrium conditions.
- **grafici_tesi.ipynb**: Jupyter notebook for generating and displaying graphical results of the simulations.

---

## Usage

To reproduce the simulations and results presented in this project, follow these steps:

### Prerequisites

- A compatible GPU
- Python 3.x
- Jupyter Notebook
- Python libraries: NumPy, CuPy, Numba

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
