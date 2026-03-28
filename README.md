# Physical Detection Limits of Dense GNSS Arrays

Code and data repository for the manuscript: 
**"Physical Detection Limits of Dense GNSS Arrays for Monitoring Pre-rupture Transient Deformation"** (Submitted to *Bulletin of the Seismological Society of America*).

## Overview
This repository contains the Python scripts and empirical noise models used to mathematically and operationally define the detection limits of a dense GNSS array ($N=1,000$). Rather than relying on idealized white-noise suppression ($\sqrt{N}$), these models explicitly incorporate the spatial covariance of urban common-mode errors (CME) and enforce a strict 120-hour causal trailing filter to ensure mechanical causality.

## Core Physical Constraints Modeled
1. **Spatial Aliasing & $N_{\mathrm{eff}}$ Asymptote:** Calculates the saturation of effective degrees of freedom due to environmental CME.
2. **Causal Notch Filtering:** Implements a strict trailing-window filter to dynamically suppress 1-cpd and 2-cpd diurnal thermal monument expansions without utilizing future data.
3. **Empirical Monte Carlo Validation:** Executes 1,000 MC realizations driven by real-world geodetic baseline noise to establish the $5\sigma$ mechanical triggering threshold ($0.87$ cm).

## Repository Structure
* `/src/01_Neff_Asymptote.py`: Derivation of the physical noise floor and spatial aliasing bounds.
* `/src/02_MonteCarlo_PoD.py`: 1,000 MC simulations generating the Probability of Detection (PoD) curve and Average Trigger Lead Time.
* `/src/03_Event_Tracking.py`: Spatiotemporal replay of 12 historical candidate transient events.
* `/src/04_Synthetic_Replay.py`: High-resolution synthetic 4D kinematic modeling of geodetic blind spots vs. dense array observability.
* `/data/`: Contains the slip parameters for the 12 historical events and empirical noise baselines.

## Dependencies
The simulations require a standard scientific Python environment.
```bash
pip install -r requirements.txt