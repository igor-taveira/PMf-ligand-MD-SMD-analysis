# PMf-ligand-MD-SMD-analysis

Python workflows for post-processing and analysis of **NAMD molecular dynamics (MD)** and **steered molecular dynamics (SMD)** simulations of small-molecule ligands interacting with an **asymmetric fungal plasma membrane (PMf)**.

The current repository includes analysis workflows developed for the ligands **MMV675968** and **MMV687807**.

## Overview

The scripts provide automated analysis of conventional MD and SMD trajectories, including:

- ligand–membrane atomic contacts;
- hydrogen-bond interactions and occupancy;
- hydrophobic contacts and occupancy;
- NAMD energetic parameters;
- bound-minus-unbound energy differences;
- SMD energetic changes;
- SMD force-versus-time analysis;
- descriptive statistics and graphical outputs;
- time-series regression and statistical inference for selected energetic parameters.

The workflows were designed for simulations performed with **NAMD** and analyzed primarily with **MDAnalysis** and the Python scientific computing ecosystem.

## Systems

### Membrane

The simulated system represents an asymmetric fungal plasma membrane (**PMf**) containing multiple lipid and membrane components, including:

- CER160
- BMAN
- DYPC
- ERG
- POPE
- POPI
- POPS
- PYPE
- YOPA
- YOPC
- YOPE

### Ligands

Currently implemented:

- **MMV675968**
- **MMV687807**

Both conventional MD trajectories and **60 Å / 1.5 ns SMD trajectories** are analyzed where applicable.

## Analyses

### Ligand–membrane contacts

Atomic contacts between the ligand and PMf are calculated frame-by-frame using a distance cutoff of:

**4.5 Å**

The scripts report contact number as a function of simulation time and generate numerical and graphical outputs.

### Hydrogen bonds

Ligand–membrane hydrogen bonds are analyzed using:

- donor–acceptor distance ≤ **3.5 Å**
- D–H···A angle ≥ **150°**

Outputs include:

- H-bonds per frame;
- overall H-bond occupancy;
- donor–acceptor pair occupancy;
- membrane-residue occupancy;
- lipid-type occupancy;
- mean H-bond distance;
- mean H-bond angle;
- raw interaction-event tables.

### Hydrophobic contacts

Hydrophobic interactions are operationally defined as contacts between ligand and membrane nonpolar carbon/sulfur atoms within:

**4.5 Å**

Outputs include:

- hydrophobic contacts per frame;
- fraction of frames containing hydrophobic interactions;
- atom-pair occupancy;
- residue occupancy;
- lipid-type occupancy;
- mean and minimum contact distances.

### NAMD energy analysis

NAMD `ENERGY:` records are automatically parsed to extract parameters such as:

- BOND
- ANGLE
- DIHED
- IMPRP
- ELECT
- VDW
- KINETIC
- POTENTIAL
- TOTAL
- temperature
- pressure
- volume

Simulation time is reconstructed from the NAMD timestep information, and scripts generate:

- complete CSV tables;
- descriptive statistics;
- individual parameter plots;
- combined energetic plots.

### Bound-minus-unbound energetic analysis

Energetic differences are calculated as:

\[
\Delta E = E_{\mathrm{bound}} - E_{\mathrm{unbound}}
\]

where the bound system corresponds to **PMf + ligand** and the unbound reference corresponds to **PMf alone**.

Equivalent comparisons are also implemented for SMD trajectories.

### SMD force analysis

SMD force records are extracted directly from NAMD output.

The scripts calculate:

- force-vector components;
- total force magnitude;
- force projected along the pulling direction;
- force versus simulation time;
- moving-average force profiles.

Force is reported in **pN**.

### Time-series statistical analysis

Selected scripts additionally perform statistical analyses designed to account for temporal dependence in MD-derived data, including:

- autocorrelation estimation;
- statistical inefficiency;
- effective sample size;
- block averaging;
- linear regression;
- autocorrelation-robust inference;
- residual analysis;
- Spearman correlation;
- multiple-testing correction.

## Main Python dependencies

The analyses use the following Python packages:

```text
numpy
pandas
matplotlib
MDAnalysis
scipy
statsmodels
