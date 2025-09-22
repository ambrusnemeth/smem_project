import numpy as np
from scipy.interpolate import interp1d
from scipy.special import kv as besselk, gamma as gamma_func
from numpy.fft import fft
from typing import Tuple, Callable
from scipy import stats

# MLE

def vg_density(x: np.ndarray, theta: float, nu: float, sigma: float, T: float) -> np.ndarray:
    if sigma <= 0 or nu <= 0:
        return np.full_like(x, np.nan)
    
    a = T / nu
    m2 = (2.0 * sigma**2) / nu + theta**2
    psi = np.sqrt(m2)
    
    v1 = 2.0 * np.exp((theta * x) / (sigma**2))
    v2 = (nu**a) * np.sqrt(2 * np.pi) * sigma * gamma_func(a)
    x_abs = np.maximum(np.abs(x), 1e-100) 
    z = (x_abs * psi) / (sigma**2)
    
    v3 = (x_abs / psi)**(a - 0.5)
    v4 = besselk(a - 0.5, z)
    
    return (v1 / v2) * v3 * v4

def neg_log_likelihood(params: Tuple[float, float, float], sample: np.ndarray, dt: float) -> float:
    theta, nu, sigma = params
    
    if sigma <= 0 or nu <= 0:
        return 1e12
        
    pdf = vg_density(sample, theta, nu, sigma, dt)
    
    if np.any(pdf <= 1e-100) or not np.all(np.isfinite(pdf)):
        return 1e12
        
    return -np.sum(np.log(pdf))

def vg_simulation(n_sim: int, n_dates: int, dt: float, params: Tuple[float, float, float]) -> np.ndarray:
    theta, nu, sigma = params
    
    g_dt = dt / nu
    
    dG = np.random.gamma(shape=g_dt, scale=nu, size=(n_dates, n_sim))
    
    dB = np.random.normal(loc=0, scale=np.sqrt(dG), size=(n_dates, n_sim))
    
    dX = theta * dG + sigma * dB
    
    return dX.flatten()

def get_initial_guess_moments(sample: np.ndarray, dt: float) -> tuple[float, float, float]:
    m = np.mean(sample)
    v = np.var(sample)
    s = stats.skew(sample)
    k = stats.kurtosis(sample, fisher=False)
    
    sigma = np.sqrt(v / dt)
    nu = (k / 3 - 1) * dt
    theta = (s * sigma * np.sqrt(dt)) / (3 * nu)

    if nu <= 0:
        print("Warning: Method of moments resulted in non-positive nu. Using a small default.")
        nu = 1e-4

    return theta, nu, sigma

# risk neutral

def phi_vg(u, S0, r, omega, T, theta, nu, sigma):
    arg = 1 - 1j * theta * nu * u + 0.5 * (sigma**2) * nu * (u**2)
    return np.exp(1j * u * (np.log(S0) + (r + omega) * T)) * (arg**(-T / nu))

def fft_pricing(T, r, char_func):
    L = 12
    N = 2**L
    dv = 0.25
    dk = 2 * np.pi / (N * dv)
    alpha = 0.75
    
    v = np.arange(N) * dv
    b = N * dk / 2
    log_strikes = -b + np.arange(N) * dk
    
    psi = np.exp(-r * T) * char_func(v - (alpha + 1) * 1j) / \
        ((alpha + 1j*v)**2 + (alpha + 1j*v))

    W = np.ones(N) * (1/3)
    W[1:-1:2] = 4/3
    W[2:-1:2] = 2/3
        
    fft_input = dv * psi * np.exp(1j * v * b) * W
    call_prices = np.exp(-alpha * log_strikes) / np.pi * np.real(fft(fft_input))
    
    return call_prices, log_strikes

def calibration_residuals(params, S0, T, r, K, P):
    theta, nu, sigma = params
    
    omega = (1 / nu) * np.log(1 - theta * nu - (sigma**2 * nu) / 2)
    
    unique_maturities = np.unique(T)
    residuals = np.zeros_like(P, dtype=float)
    
    for t in unique_maturities:
        idx = (T == t)
        
        cf = lambda w: phi_vg(w, S0, r, omega, t, theta, nu, sigma)
        
        call_prices_fft, log_strikes_fft = fft_pricing(t, r, cf)
        
        interp_func = interp1d(log_strikes_fft, call_prices_fft, kind='linear', fill_value="extrapolate")
        
        selected_strikes = K[idx]
        interpolated_prices = interp_func(np.log(selected_strikes))
        
        residuals[idx] = (interpolated_prices - P[idx])
        
    return residuals
