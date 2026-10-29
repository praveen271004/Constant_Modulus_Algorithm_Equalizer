
import numpy as np
import matplotlib.pyplot as plt

from cma_equalizer import cma_equalize

# 1. Generate reproducible QPSK symbols
rng = np.random.default_rng(42)
N = 10000

indices = rng.integers(0, 4, size=N)

constellation = np.array([
    1 + 1j,
    -1 + 1j,
    -1 - 1j,
    1 - 1j
]) / np.sqrt(2)

symbols = constellation[indices]

# 2. Simulate a multipath channel
channel = np.array([1.0, 0.45, 0.2])
received_clean = np.convolve(
    symbols, channel, mode="full"
)[:N]

# 3. Add complex Gaussian noise
noise_power = 0.02
noise = np.sqrt(noise_power / 2) * (
    rng.standard_normal(N)
    + 1j * rng.standard_normal(N)
)

received = received_clean + noise

# 4. Run CMA equalization
equalized, cost, weights = cma_equalize(
    received,
    num_taps= nine if False else 9,
    mu=0.001,
    R=1.0
)

# 5. Plot constellation diagrams
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

axes[0].scatter(
    symbols.real, symbols.imag, s=7, alpha=0.5
)
axes[0].set_title("Transmitted QPSK")

axes[1].scatter(
    received.real, received.imag, s=7, alpha=0.5
)
axes[1].set_title("Received Before CMA")

axes[2].scatter(
    equalized[500:].real,
    equalized[500:].imag,
    s=7, alpha=0.5
)
axes[2].set_title("Output After CMA")

for ax in axes:
    ax.set_xlabel("In-phase (I)")
    ax.set_ylabel("Quadrature (Q)")
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")

plt.tight_layout()
plt.show()

# 6. Plot smoothed CMA cost
window = 100
smoothed_cost = np.convolve(
    cost, np.ones(window) / window, mode="valid"
)

plt.figure(figsize=(9, 4))
plt.plot(smoothed_cost)
plt.xlabel("Sample index")
plt.ylabel("Mean CMA cost")
plt.title("CMA Convergence")
plt.grid(True)
plt.tight_layout()
plt.show()

# 7. Display final coefficients
print("Final CMA coefficients:")
print(weights)

print("Initial average CMA cost:", np.mean(cost[:500]))
print("Final average CMA cost:", np.mean(cost[-500:]))



# 8. Quantitative validation

# Ignore the initial transient when evaluating steady-state behavior.
skip = 500
steady_output = equalized[skip:]
steady_cost = cost[skip:]

initial_cost = np.mean(cost[:500])
final_cost = np.mean(cost[-500:])

# The CMA objective uses squared magnitude.
magnitude_error = np.mean(
    (np.abs(steady_output) ** 2 - 1.0) ** 2
)

print("\n--- CMA Validation ---")
print(f"Initial average cost: {initial_cost:.6f}")
print(f"Final average cost:   {final_cost:.6f}")
print(f"Steady-state magnitude cost: {magnitude_error:.6f}")

if initial_cost > 0:
    reduction = 100 * (initial_cost - final_cost) / initial_cost
    print(f"Cost reduction: {reduction:.2f}%")

# Offline SER evaluation.
# Search for the best integer delay and QPSK phase rotation.
# This does NOT affect the blind CMA coefficient updates.

best_ser = 1.0
best_delay = None
best_phase = None

for delay in range(-20, 21):

    if delay >= 0:
        y = equalized[delay + skip:]
        s = symbols[:len(y)]
    else:
        y = equalized[skip:len(equalized) + delay]
        s = symbols[-delay:len(symbols) + delay]

    length = min(len(y), len(s))
    y = y[:length]
    s = s[:length]

    if length == 0:
        continue

    # Estimate the phase ambiguity using reference symbols.
    phase = np.angle(np.sum(y * np.conj(s)))

    # Compensate phase and normalize average magnitude.
    y_corrected = y * np.exp(-1j * phase)
    scale = np.mean(np.abs(y_corrected))

    if scale < 1e-12:
        continue

    y_corrected = y_corrected / scale

    # Choose the nearest ideal QPSK constellation point.
    distances = np.abs(
        y_corrected[:, None] - constellation[None, :]
    )
    decisions = constellation[np.argmin(distances, axis=1)]

    # The reference symbols are used only for evaluation.
    ser = np.mean(decisions != s)

    if ser < best_ser:
        best_ser = ser
        best_delay = delay
        best_phase = phase

print(f"Best evaluation delay: {best_delay}")
print(f"Estimated phase offset: {best_phase}")
print(f"Best aligned SER: {best_ser:.6f}")
