# V6.20 Rhombohedral Nanocube Lattice Calculation

This directory contains the V6.20 calculation and its archived outputs. The model evaluates the energy of magnetic Fe3O4 nanocubes in a rhombohedral superlattice. The lattice angle, common particle rotation, and lattice constant are optimized at fixed nearest surface separation.

## Model

The energy per particle contains Zeeman, cubic magnetocrystalline anisotropy, dipolar, and van der Waals contributions. The dipolar term is summed over the 26 neighbors of the central particle in a 3 x 3 x 3 lattice shell. The van der Waals term uses the six primitive neighbors. Each cube is represented by a 4 x 4 x 4 voxel array for the van der Waals calculation.

The 16 nm calculation uses a saturation magnetization of 2.85 x 10^5 A m^-1, an applied field of 0.50 T, an anisotropy constant of 2.00 x 10^4 J m^-3, a Hamaker constant of 2.00 x 10^-20 J, a rounding radius of 1.5 nm, and a temperature of 298.15 K.

## Files

`code/V6.20_energy_landscape_EaAlpha/V6.20rhombo_EaAlpha.ipynb` contains the lattice energy calculation and structural minimization.

`code/V6.20_energy_landscape_EaAlpha/v620_two_nc_temperature.py` contains the two particle temperature and kinetic calculations.

`outputs/V6.20_energy_landscape_EaAlpha/results_archive/` contains the archived numerical data and figures.

## Requirements

- Python 3.10 or later
- NumPy
- SciPy
- Matplotlib
- JupyterLab or Jupyter Notebook

Install the required packages with

```bash
python -m pip install numpy scipy matplotlib jupyterlab
```

## Usage

Open the lattice calculation with

```bash
jupyter lab code/V6.20_energy_landscape_EaAlpha/V6.20rhombo_EaAlpha.ipynb
```

Run the temperature calculation with

```bash
python code/V6.20_energy_landscape_EaAlpha/v620_two_nc_temperature.py
```

The temperature script writes new files to `outputs/V6.20_energy_landscape_EaAlpha/generated/`.

## 16 nm Result

For the fixed gap 16 nm calculation, the minimum occurs at alpha = 74.177 degrees, gamma = -16.517 degrees, and a = 20.1963 nm. The dipolar and van der Waals contributions are -3.5877 kBT and -2.2559 kBT. The interaction energy is -5.8436 kBT per particle.

The numerical values are stored in `outputs/V6.20_energy_landscape_EaAlpha/results_archive/V620_EaAlpha/V620_size_minima.csv`.

## References

1. Faure, B., Salazar-Alvarez, G., Bergstrom, L. Hamaker Constants of Iron Oxide Nanoparticles. *Langmuir* **2011**, *27*, 8659-8664. https://doi.org/10.1021/la201387d
2. Hamaker, H. C. The London van der Waals Attraction between Spherical Particles. *Physica* **1937**, *4*, 1058-1072. https://doi.org/10.1016/S0031-8914(37)80203-7

## Citation

If this code is used in a publication, cite the associated article and this repository. Publication details can be added after acceptance.
