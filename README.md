# Blind Adaptive Channel Equalization Using the Constant Modulus Algorithm (CMA)

## Overview

This project implements a **blind adaptive equalizer using the Constant Modulus Algorithm (CMA)** in Python. It simulates a digital communication system in which transmitted symbols pass through a multipath channel and are corrupted by complex Gaussian noise. The CMA equalizer then attempts to compensate for channel-induced distortion without using known transmitted symbols during coefficient adaptation.

The project evaluates the equalizer using constellation diagrams, CMA cost curves, and an offline symbol error rate (SER) comparison.

## Objectives

- Generate digitally modulated complex symbols.
- Model inter-symbol interference (ISI) using a multipath channel.
- Add complex Gaussian noise to the received signal.
- Implement a tap-based adaptive CMA equalizer.
- Observe equalizer convergence through the CMA cost.
- Compare performance across different modulation schemes.
- Estimate SER after offline delay and complex-gain alignment.

## How It Works

### 1. Symbol generation

The simulation generates random symbols from a selected modulation constellation. The experiments include:

- BPSK
- QPSK
- 8-PSK
- 16-QAM
- 64-QAM

The PSK constellations have constant symbol magnitude. The QAM constellations have multiple symbol magnitudes and are normalized to unit average symbol power.

### 2. Multipath channel

The transmitted sequence passes through a finite impulse response (FIR) channel:

\[
h = [1.0,\;0.45,\;0.2]
\]

The channel output is computed by convolution. The delayed channel components introduce inter-symbol interference (ISI).

### 3. Noise addition

Complex Gaussian noise is added to the channel output. The simulation uses a noise power of `0.02` and a fixed random seed by default, making experiments reproducible.

### 4. CMA equalization

The equalizer uses a configurable number of taps and updates its complex-valued coefficients sample by sample.

For equalizer input vector \(x_n\), coefficient vector \(w_n\), and output \(y_n\):

\[
y_n = w_n^H x_n
\]

The CMA error term is:

\[
e_n = R - |y_n|^2
\]

The instantaneous cost is:

\[
J_n = (R - |y_n|^2)^2
\]

The coefficient update implemented in this project is:

\[
w_{n+1} = w_n + \mu x_n y_n^* (R - |y_n|^2)
\]

where:

- \(w_n\) is the equalizer coefficient vector.
- \(x_n\) is the received-sample vector.
- \(y_n\) is the equalizer output.
- \(\mu\) is the adaptation step size.
- \(R\) is the target squared output magnitude.

For the multi-modulation experiment, the dispersion constant is calculated as:

\[
R = \frac{\mathbb{E}[|a|^4]}{\mathbb{E}[|a|^2]}
\]

where \(a\) represents constellation symbols. This gives a modulation-dependent target for QAM as well as PSK.

**Important:** CMA is a blind adaptation method: transmitted reference symbols are not used in the coefficient update. Reference symbols are used later by the simulation to evaluate SER.

## Repository Structure

```text
.
├── main.py                  # QPSK simulation, CMA equalization, plots and validation
├── cma_equalizer.py         # Reusable CMA equalizer implementation
├── multiple_mod_schemes.py  # Experiments for BPSK, QPSK, 8-PSK, 16-QAM and 64-QAM
├── AI_log.md                # AI usage and verification log
└── README.md                # Project documentation
```

## Requirements

- Python 3.9 or later recommended
- NumPy
- Matplotlib

## Installation

1. Install Python if it is not already installed.
2. Clone or download this repository.
3. Open a terminal in the project directory.
4. Install the dependencies:

```bash
python -m pip install numpy matplotlib
```

## Running the Project

### QPSK experiment

Run the main QPSK simulation:

```bash
python main.py
```

This script generates QPSK symbols, applies the multipath channel and noise, runs CMA equalization, and displays constellation and convergence plots. It also prints the final coefficients and several validation metrics.

### Multiple-modulation experiment

Run the comparison across modulation schemes:

```bash
python multiple_mod_schemes.py
```

This script runs the experiment for BPSK, QPSK, 8-PSK, 16-QAM and 64-QAM. It prints a comparison table and plots received/equalized constellations, smoothed CMA cost, and SER.

Run commands from the directory containing all three Python files so the import `from cma_equalizer import cma_equalize` can be resolved.

## Parameters You Can Experiment With

The following parameters can be adjusted in the scripts:

| Parameter | Purpose |
|---|---|
| `N` | Number of transmitted symbols |
| `channel` | Multipath channel coefficients |
| `noise_power` | Noise power |
| `num_taps` | Number of equalizer taps |
| `mu` | CMA adaptation step size |
| `R` | Target squared output magnitude / dispersion constant |
| `seed` | Random seed for reproducibility |

Change one parameter at a time and compare convergence, output constellations and SER. A larger step size may adapt faster but can also make the algorithm unstable; a smaller step size may converge more slowly.

## Outputs and Evaluation

The scripts provide the following outputs:

- **Transmitted constellation:** Ideal symbols before channel distortion.
- **Received constellation:** Symbols after multipath distortion and noise.
- **Equalized constellation:** Equalizer output after adaptation.
- **CMA cost curve:** Instantaneous or smoothed cost over the sample sequence.
- **Initial and final average cost:** Used to inspect changes in the objective during adaptation.
- **Symbol error rate (SER):** Computed after searching for a suitable delay and applying an offline complex-gain correction.

A lower final CMA cost indicates that the output magnitudes are closer to the selected target on average. It does **not**, by itself, guarantee a low SER or perfect recovery of the transmitted symbols. The CMA objective is non-convex and performance depends on the channel, modulation, step size, initialization and other settings.

The SER calculation is an offline evaluation procedure using the known transmitted sequence. It does not feed decisions or reference symbols back into the CMA coefficient updates.

## Limitations

- The channel and noise are simulated; this repository does not process live radio hardware or real recorded signals.
- The provided experiments use a fixed channel and fixed noise-power setting unless the source code is changed.
- CMA is naturally suited to constant-modulus constellations such as PSK. Its behavior with QAM constellations can be less direct because QAM symbols have different magnitudes.
- Blind equalization can exhibit phase ambiguity and decision errors. The scripts include offline alignment for evaluation, not a complete carrier-recovery or timing-recovery receiver.
- Results depend on the selected parameters and should be interpreted from both the cost curves and SER metrics.

## Suggested Experiments

1. Change `mu` and compare convergence speed and stability.
2. Vary `num_taps` to investigate equalizer length.
3. Increase the noise power and observe the constellation spread and SER.
4. Change the multipath channel coefficients to create different levels of ISI.
5. Compare the CMA cost and SER for PSK and QAM schemes.
6. Repeat experiments with different random seeds to assess result variability.

## AI Usage Log

`AI_log.md` is included to document AI assistance, verification steps, changes made after local testing, and limitations. Update the log to reflect what you actually tested and changed before submitting or publishing the repository.

## License

No license is currently specified. Add a `LICENSE` file if you want to define the terms under which others may use, modify or distribute this project.

## Author

Add your name, institution (if desired), and contact or profile link here.
