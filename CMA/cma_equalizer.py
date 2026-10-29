
import numpy as np

def cma_equalize(received, num_taps=9, mu=0.001, R=1.0):

    """
    Blind CMA equalizer for complex-valued signals.

    Parameters
    ----------
    received : array-like
        Complex received samples.
    num_taps : int
        Number of FIR equalizer coefficients.
    mu : float
        Adaptation step size.
    R : float
        Desired squared output magnitude.

    Returns
    -------
    output : ndarray
        Equalizer output samples.
    cost : ndarray
        Instantaneous CMA cost.
    weights : ndarray
        Final equalizer coefficients.
    """

    received = np.asarray(received, dtype=complex)

    if num_taps < 1:
        raise ValueError("num_taps must be at least 1")

    if mu <= 0:
        raise ValueError("mu must be positive")

    # Initialize coefficients to zero.
    weights = np.zeros(num_taps, dtype=complex)

    # Start with the center tap enabled.
    weights[num_taps // 2] = 1.0

    # Pad the signal to support the initial filter windows.
    padded = np.pad(received, (num_taps - 1, 0))

    output = np.zeros(len(received), dtype=complex)
    cost = np.zeros(len(received))

    for n in range(len(received)):

        # Current sample followed by previous samples.
        x_vec = padded[n:n + num_taps][::-1]

        # Equalizer output: y = w^H x.
        y = np.vdot(weights, x_vec)

        # CMA magnitude error.
        error = R - abs(y) ** 2

        # Record output and cost before coefficient update.
        output[n] = y
        cost[n] = error ** 2

        # Stochastic-gradient coefficient update.
        weights += mu * x_vec * np.conj(y) * error

    return output, cost, weights
