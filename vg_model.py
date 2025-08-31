import numpy as np
from scipy.special import kv as besselk, gamma
from numpy.fft import fft

# ---------- PDF (historical MLE) ----------
def VGdensity(x, theta, nu, sigma, T):
    x = np.asarray(x, dtype=float)
    a = T/nu
    if sigma <= 0 or nu <= 0: return np.full_like(x, np.nan)
    M2  = (2.0*sigma**2)/nu + theta**2
    psi = np.sqrt(M2)
    z   = (np.abs(x)*psi)/(sigma**2)
    v1  = 2.0*np.exp((theta*x)/(sigma**2)) / ((nu**a)*np.sqrt(2*np.pi)*sigma*gamma(a))
    v4  = (np.maximum(np.abs(x), 1e-300)/psi)**(a-0.5)
    K   = besselk(a-0.5, np.maximum(z, 1e-300))
    return v1*v4*K

def neg_loglike(params, sample, dt):
    theta, nu, sigma = params
    if sigma <= 0 or nu <= 0: return 1e12
    pdf = VGdensity(sample, theta, nu, sigma, dt)
    if np.any(pdf <= 0) or not np.all(np.isfinite(pdf)): return 1e12
    return -np.sum(np.log(pdf))

# ---------- Simulation ----------
def VG_simulation(Nsim, nDates, dt, params):
    theta, nu, sigma = params
    g = np.random.gamma(dt/nu, nu, size=(Nsim, nDates))
    X = np.zeros((Nsim, nDates))
    for j in range(1, nDates):
        X[:, j] = X[:, j-1] + theta*g[:, j] + sigma*np.sqrt(g[:, j])*np.random.randn(Nsim)
    return X

# ---------- RN drift fix (used by both H & RN parameter sets) ----------
def omega_vg(theta, nu, sigma):
    return (1.0/nu)*np.log(1.0 - theta*nu - 0.5*sigma*sigma*nu)

# ---------- VG CF under Black-76 (NO r in the exponent) ----------
def phi_vg_b76(u, F0, omega, T, theta, nu, sigma):
    u = np.asarray(u, dtype=complex)
    argg = 1.0 - 1j*theta*nu*u + 0.5*sigma**2 * nu * u**2
    return np.exp(1j*u*(np.log(F0) + omega*T)) * (argg**(-T/nu))

# ---------- Carr–Madan FFT (kept as in MATLAB) ----------
def FFTPricing(T, r, phi):
    L, dv, alpha = 12, 0.25, 0.75
    N  = 2**L
    dk = 2*np.pi/(N*dv)
    v  = (np.arange(1, N+1) - 1) * dv
    b  = (N*dk)/2.0
    ku = -b + dk*(np.arange(1, N+1) - 1)

    psi = np.exp(-r*T) * phi(v - 1j*(alpha+1.0)) / (alpha**2 + alpha - v**2 + 1j*(2*alpha+1.0)*v)

    W = np.empty(N)
    W[0] = 1/3; W[1::2] = 4/3; W[2::2] = 2/3
    tmp = dv * psi * np.exp(1j*v*b) * W
    CallPrices = np.exp(-alpha*ku) / np.pi * np.real(fft(tmp))
    return CallPrices, ku

# ---------- Price slice (Black-76) ----------
def price_slice_fft_b76(F0, r, T, theta, nu, sigma, Ks_query):
    om  = omega_vg(theta, nu, sigma)
    cf  = lambda w: phi_vg_b76(w, F0, om, T, theta, nu, sigma)  # NOTE: no r here
    C, logK = FFTPricing(T, r, cf)                              # discount in FFTPricing
    return np.interp(np.log(np.asarray(Ks_query)), logK, C)

# ---------- Residual vector (squared) for lsqnonlin-style use ----------
def resid_vector_squared(x, F0, T, r, K, P):
    theta, nu, sigma = x
    T = np.asarray(T); K = np.asarray(K); P = np.asarray(P)
    out = np.empty_like(P, dtype=float)
    pos = 0
    for Tmat in np.unique(T):
        idx = (T == Tmat)
        model = price_slice_fft_b76(F0, r, Tmat, theta, nu, sigma, K[idx])
        out[idx] = (model - P[idx])**2  # squared residuals (to mimic your MATLAB)
        pos += idx.sum()
    return out