"""
This module contain a few classes to quickly sample from a PDF (could be
tabulated or a function).
It creates a tabulated CDF to map from [0,1] → [a,b] e.g. the PDF domain.

Here we also store pdf functions that can be used for fitting.
"""
import numpy as np

class PDFSampler:
    """
    Precompute a tabulated CDF on [a,b] from a non-negative PDF function, then
    expose fast methods:
      - ppf(u): inverse CDF (maps uniform u in [0,1) to x in [a,b])
      - cdf(x): CDF evaluation (x -> [0,1])
      - sample(size, rng): draw samples via inverse transform

    Input:
        pdf_func : callable. PDF function defined within a,b.
            Note that the pdf_func input should assume the input distance have
            amuse units.
        limits   : tuple, min and max domain range for the pdf
        nsample  : number of sample points to sample pdf_func

    Note: This module do not use amuse units. Make sure all quantities are
    passed with in the correct units of the pdf_func.
    """
    def __init__(self, pdf_func, limits, nsample=4096):
        self.limits = limits
        self.nsample = int(nsample)
        self.n = self.nsample  # internal convenience

        # Tabulate x
        x = np.linspace(limits[0], limits[1], self.n).astype(float)
        p = pdf_func(x)

        # Make sure pdf evaluation is non-negative
        if np.any(p < 0):
            raise ValueError('pdf_func evaluations give negative values')

        # Build cumulative integral using trapezoid rule
        c = np.empty_like(p, dtype=float)
        c[0] = 0.0
        if self.n > 1:
            dx = np.diff(x)
            c[1:] = np.cumsum(0.5 * (p[1:] + p[:-1]) * dx)

        total = c[-1]
        if not np.isfinite(total) or total <= 0:
            raise ValueError("PDF integrates to ~0 over [a,b]; widen domain or fix pdf.")
        c /= total  # normalize to [0,1]

        # Enforce strict monotonicity (protect against flat segments)
        for i in range(1, self.n):
            if c[i] <= c[i-1]:
                c[i] = np.nextafter(c[i-1], 1.0)

        # Keep arrays for lookups
        self.x = x
        self.c = c

    def ppf(self, u):
        """Inverse CDF: map u in [0,1) to x in [a,b] using linear segments."""
        u = np.asarray(u, float)
        u = np.clip(u, 0.0, 1.0 - 1e-12)
        idx = np.searchsorted(self.c, u, side="right")
        idx = np.clip(idx, 1, self.n - 1)
        c0 = self.c[idx - 1]; c1 = self.c[idx]
        x0 = self.x[idx - 1]; x1 = self.x[idx]
        t = np.where(c1 > c0, (u - c0) / (c1 - c0), 0.0)
        return x0 + t * (x1 - x0)

    def cdf(self, xq):
        """CDF evaluation at xq via linear interpolation on the tabulated grid."""
        xq = np.asarray(xq, float)
        xi = np.clip(xq, self.x[0], self.x[-1])
        idx = np.searchsorted(self.x, xi, side="right")
        idx = np.clip(idx, 1, self.n - 1)
        x0 = self.x[idx - 1]; x1 = self.x[idx]
        c0 = self.c[idx - 1]; c1 = self.c[idx]
        t = np.where(x1 > x0, (xi - x0) / (x1 - x0), 0.0)
        out = c0 + t * (c1 - c0)
        out = np.where(xq < self.x[0], 0.0, out)
        out = np.where(xq >= self.x[-1], 1.0, out)
        return out

    def sample(self, size=1, rng=None):
        """Draw samples by inverse transform using precomputed PPF."""
        g = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        u = g.random(size)
        return self.ppf(u)

def lognormal_pdf(r, mu=-2.15, sigma=0.9):
    """Lognormal PDF in natural-log space; r must be > 0 (same length units as positions)."""
    r = np.asarray(r, float)
    r = np.maximum(r, 1e-12)
    return (1.0 / (r * sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((np.log(r) - mu) / sigma) ** 2)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.special import erf
    from math import sqrt

    erf_vec = np.vectorize(erf)

    # Lognormal params and domain that captures ~all mass
    mu, sigma = -2.15, 0.9
    lo = float(np.exp(mu - 5 * sigma))
    hi = float(np.exp(mu + 5 * sigma))
    limits = (lo, hi)

    # Build sampler from the lognormal PDF
    sampler = PDFSampler(lambda r: lognormal_pdf(r, mu, sigma), limits, nsample=4096)

    # x-grid (log-spaced suits a lognormal)
    xs = np.logspace(np.log10(limits[0]), np.log10(limits[1]), 1000)

    # Analytic PDF and CDF
    pdf_curve = lognormal_pdf(xs | units.parsec, mu, sigma)
    cdf_curve = 0.5 * (1.0 + erf_vec((np.log(xs) - mu) / (sigma * sqrt(2.0))))

    # Samples: small and large to compare noise vs convergence
    s_small = sampler.sample(1_000, rng=123)
    s_big   = sampler.sample(200_000, rng=42)

    # Common log bins for fair PDF comparison
    bins = np.logspace(np.log10(limits[0]), np.log10(limits[1]), 80)

    # Empirical CDF helper
    def ecdf(samples):
        y = np.sort(samples)
        n = y.size
        x = np.r_[y[0] * 0.9999, y]        # start just below min for a nicer step
        F = np.r_[0.0, np.arange(1, n + 1) / n]
        return x, F

    ex_small_x, ex_small_F = ecdf(s_small)
    ex_big_x,   ex_big_F   = ecdf(s_big)

    # ---- Plot: side-by-side PDF and CDF ----
    fig, (ax_pdf, ax_cdf) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: PDF
    ax_pdf.hist(s_small, bins=bins, density=True, alpha=0.35, label="Samples (N=1k)")
    ax_pdf.hist(s_big,   bins=bins, density=True, alpha=0.25, label="Samples (N=200k)")
    ax_pdf.plot(xs, pdf_curve, lw=2.0, label="Analytic lognormal PDF")
    ax_pdf.set_xscale("log")
    ax_pdf.set_yscale("log")
    ax_pdf.set_xlabel("r")
    ax_pdf.set_ylabel("PDF")
    ax_pdf.legend()
    ax_pdf.set_title("PDF: analytic vs samples")

    # Right: CDF
    ax_cdf.plot(xs, cdf_curve, "--", lw=2.0, label="Analytic CDF")
    ax_cdf.step(ex_small_x, ex_small_F, where="post", alpha=0.8, label="Empirical CDF (N=1k)")
    ax_cdf.step(ex_big_x,   ex_big_F,   where="post", alpha=0.6, label="Empirical CDF (N=200k)")
    ax_cdf.set_xscale("log")
    ax_cdf.set_ylim(0.0, 1.0)
    ax_cdf.set_xlabel("r")
    ax_cdf.set_ylabel("CDF")
    ax_cdf.legend()
    ax_cdf.set_title("CDF: analytic vs samples")

    fig.tight_layout()

    plt.savefig('pdfsampler_test.pdf')
