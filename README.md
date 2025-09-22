# Variance Gamma Model Calibration for Electricity Forwards and Options

This repository contains my exam project for the course  
[**Stochastic Methods of Energy Markets**](https://studies.helsinki.fi/courses/course-implementation/hy-opt-cur-2425-462a39f0-a0c1-4367-acc8-130564ddd365)  
at the **University of Helsinki**.

The methodology is based on a lecture given by Matteo Gardini at the University of Helsinki in February 2025. 

---

## Project description

The aim of the project is to **calibrate the Variance Gamma (VG) process** to German power forward prices and option quotes on the DEBY 2021.01 contract.

- **Historical calibration:**  
  Daily forward prices are used to estimate VG parameters \((\theta, \sigma, \nu)\) by maximum likelihood (MLE).  
  This part highlights how the VG process captures skewness and heavy tails in log-returns.

- **Risk-neutral calibration:**  
  Quoted option prices (calls on the same forward) are used to re-calibrate the parameters by least squares.  
  Pricing is done with the **Carr–Madan FFT** implementation of Black–76.  
  The results compare market prices with both historical and risk-neutral fits.

---

## Repository contents

- `Historical_Prices_FWD_Germany.csv` — Daily settlement prices for DEBY 2021.01 forwards, used for historical calibration.  
- `Options_Prices_Calendar_2021.csv` — Quoted option prices on the same forward, used for risk-neutral calibration.  
- `Forward_Prices.csv` — Additional forward data for DEBY contracts.  
- `Modeling German Power Forwards with the Variance Gamma Process.ipynb` — Main Jupyter notebook with full analysis.  
- `Modeling German Power Forwards with the Variance Gamma Process.pdf` — PDF export of the notebook.  
- `math_formulas.py` — Core implementation of the VG model (density, simulation, FFT pricing, calibration residuals).  
- `data_loading.py` — Helper script for loading and preprocessing CSV data.  
- `vg_energy_markets_helsinki.pdf` — Slides of Matteo Gardini’s guest lecture (reference material).
