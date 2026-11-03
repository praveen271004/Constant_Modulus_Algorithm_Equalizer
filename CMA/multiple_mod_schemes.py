
import numpy as np
import matplotlib.pyplot as plt

from cma_equalizer import cma_equalize


def make_constellation(modulation):
    if modulation == "BPSK":
        constellation = np.array([-1, 1], dtype=complex)

    elif modulation == "QPSK":
        constellation = np.array([
            1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j
        ], dtype=complex) / np.sqrt(2)

    elif modulation == "8-PSK":
        phases = 2 * np.pi * np.arange(8) / 8
        constellation = np.exp(1j * phases)

    elif modulation in ("16-QAM", "64-QAM"):
        order = 4 if modulation == "16-QAM" else 8
        levels = np.arange(-(order - 1), order, 2)
        i_values, q_values = np.meshgrid(levels, levels)
        constellation = (
            i_values.ravel() + 1j * q_values.ravel()
        ).astype(complex)

        # Normalize average symbol power to one.
        constellation /= np.sqrt(
            np.mean(np.abs(constellation) ** 2)
        )

    else:
        raise ValueError(f"Unsupported modulation: {modulation}")

    return constellation


def evaluate_ser(output, symbols, constellation, skip=500,
                 max_delay=20):
    best_ser = 1.0
    best_delay = None

    for delay in range(-max_delay, max_delay + 1):
        start = max(skip, delay, 0)
        end = min(len(output), len(symbols) + delay)

        if end <= start:
            continue

        y = output[start:end]
        reference = symbols[start - delay:end - delay]

        # Estimate complex gain for offline evaluation only.
        denominator = np.sum(np.abs(y) ** 2)

        if denominator < 1e-12:
            continue

        gain = np.sum(reference * np.conj(y)) / denominator
        corrected = gain * y

        distances = np.abs(
            corrected[:, None] - constellation[None, :]
        )
        decisions = constellation[np.argmin(distances, axis=1)]

        ser = np.mean(decisions != reference)

        if ser < best_ser:
            best_ser = ser
            best_delay = delay

    return best_ser, best_delay


def run_experiment(modulation, seed=42):
    rng = np.random.default_rng(seed)

    N = 10000
    constellation = make_constellation(modulation)

    # Generate random symbols.
    indices = rng.integers(0, len(constellation), size=N)
    symbols = constellation[indices]

    # Apply the same multipath channel to every modulation.
    channel = np.array([1.0, 0.45, 0.2])
    clean = np.convolve(symbols, channel, mode="full")[:N]

    # Add equal average-power complex noise.
    noise_power = 0.02
    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal(N)
        + 1j * rng.standard_normal(N)
    )
    received = clean + noise

    # CMA dispersion constant:
    # R = E[|a|^4] / E[|a|^2].
    # This is 1 for normalized PSK, but greater than 1 for QAM.
    R = (
        np.mean(np.abs(constellation) ** 4)
        / np.mean(np.abs(constellation) ** 2)
    )

    equalized, cost, weights = cma_equalize(
        received,
        num_taps=9,
        mu=0.001,
        R=R
    )

    skip = 500
    ser, delay = evaluate_ser(
        equalized, symbols, constellation, skip=skip
    )

    result = {
        "modulation": modulation,
        "symbols": symbols,
        "received": received,
        "equalized": equalized,
        "cost": cost,
        "R": R,
        "ser": ser,
        "delay": delay,
        "initial_cost": np.mean(cost[:500]),
        "final_cost": np.mean(cost[-500:]),
        "weights": weights
    }

    return result


# Run the same CMA algorithm for all five modulation schemes.
modulations = ["BPSK", "QPSK", "8-PSK", "16-QAM", "64-QAM"]
results = []

for modulation in modulations:
    result = run_experiment(modulation)
    results.append(result)

# Print a comparison table.
print("\nCMA MULTI-MODULATION RESULTS")
print("-" * 82)
print(
    f"{'Modulation':<12} {'R':>8} {'Initial Cost':>14} "
    f"{'Final Cost':>14} {'SER':>12} {'Delay':>8}"
)
print("-" * 82)

for result in results:
    print(
        f"{result['modulation']:<12} "
        f"{result['R']:>8.4f} "
        f"{result['initial_cost']:>14.6f} "
        f"{result['final_cost']:>14.6f} "
        f"{result['ser']:>12.6f} "
        f"{str(result['delay']):>8}"
    )

# Plot received and equalized constellations.
fig, axes = plt.subplots(5, 2, figsize=(11, 20))

for i, result in enumerate(results):
    received = result["received"]
    equalized = result["equalized"]
    symbols = result["symbols"]

    axes[i, 0].scatter(
        received.real, received.imag, s=5, alpha=0.4
    )
    axes[i, 0].set_title(
        f"{result['modulation']} - Before CMA"
    )

    axes[i, 1].scatter(
        equalized[500:].real,
        equalized[500:].imag,
        s=5, alpha=0.4
    )
    axes[i, 1].scatter(
        symbols.real[:500],
        symbols.imag[:500],
        s=12, marker="x"
    )
    axes[i, 1].set_title(
        f"{result['modulation']} - After CMA"
    )

    for ax in axes[i]:
        ax.set_xlabel("In-phase (I)")
        ax.set_ylabel("Quadrature (Q)")
        ax.grid(True)
        ax.set_aspect("equal", adjustable="box")

plt.tight_layout()
plt.show()

# Compare smoothed CMA cost.
plt.figure(figsize=(10, 5))

window = 100

for result in results:
    cost = result["cost"]
    smoothed = np.convolve(
        cost, np.ones(window) / window, mode="valid"
    )
    plt.plot(smoothed, label=result["modulation"])

plt.xlabel("Sample index")
plt.ylabel("Smoothed CMA cost")
plt.title("CMA Convergence for Different Modulations")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Compare final SER.
plt.figure(figsize=(9, 5))

plt.bar(
    [result["modulation"] for result in results],
    [result["ser"] for result in results]
)

plt.xlabel("Modulation")
plt.ylabel("Best aligned symbol error rate")
plt.title("CMA Symbol Error Rate Comparison")
plt.grid(axis="y")
plt.tight_layout()
plt.show()
