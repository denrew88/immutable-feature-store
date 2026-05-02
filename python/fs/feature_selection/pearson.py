import numpy as np


def _sanitize_values_and_mask(values, valid):
    """Zero-fill NaNs and clear validity bits at the same positions."""
    if values is None or valid is None:
        raise ValueError("values and valid must not be None")
    if values.shape != valid.shape:
        raise ValueError("values and valid must have the same shape")

    if np.isnan(values).any():
        values = values.copy()
        nan_mask = np.isnan(values)
        values[nan_mask] = 0.0
        valid = valid.copy()
        valid[nan_mask] = 0
    return values, valid


def pairwise_r2(x, mx, z, mz, min_non_null=2):
    """Compute exact masked r^2 for one feature pair."""
    if x.shape != z.shape or mx.shape != mz.shape:
        raise ValueError("x/z and mx/mz shapes must match")
    m = (mx.astype(bool) & mz.astype(bool))
    if np.isnan(x).any():
        m = m & ~np.isnan(x)
    if np.isnan(z).any():
        m = m & ~np.isnan(z)
    n = int(m.sum())
    if n < max(2, min_non_null):
        return 0.0, n
    xv = x[m]
    zv = z[m]
    sum_x = xv.sum()
    sum_z = zv.sum()
    sum_x2 = (xv * xv).sum()
    sum_z2 = (zv * zv).sum()
    sum_xz = (xv * zv).sum()
    mx_v = sum_x / n
    mz_v = sum_z / n
    var_x = sum_x2 - n * mx_v * mx_v
    var_z = sum_z2 - n * mz_v * mz_v
    if var_x <= 0.0 or var_z <= 0.0:
        return 0.0, n
    cov = sum_xz - n * mx_v * mz_v
    r = cov / np.sqrt(var_x * var_z)
    return float(r * r), n


def batch_r2_one_vs_many(x, mx, Z, Mz, min_non_null=2, sanitize=False):
    """Compute masked r^2 between one vector and many candidate rows."""
    if Z.ndim != 2 or Mz.ndim != 2:
        raise ValueError("Z and Mz must be 2D arrays")
    if Z.shape != Mz.shape:
        raise ValueError("Z and Mz shapes must match")
    if x.ndim != 1 or mx.ndim != 1:
        raise ValueError("x and mx must be 1D arrays")
    if Z.shape[1] != x.shape[0]:
        raise ValueError("Z columns must match x length")

    if sanitize:
        Z, Mz = _sanitize_values_and_mask(Z, Mz)

    mx_f = mx.astype(np.float64, copy=False)
    x = x.astype(np.float64, copy=False)
    if np.isnan(x).any():
        mx_f = mx_f * (~np.isnan(x)).astype(np.float64)
        x = np.nan_to_num(x, nan=0.0)
    x_f = x * mx_f
    x2 = x_f * x_f

    n = (Mz @ mx_f).astype(np.int64, copy=False)
    sum_x = Mz @ x_f
    sum_z = Z @ mx_f
    sum_x2 = Mz @ x2
    sum_z2 = (Z * Z) @ mx_f
    sum_xz = Z @ x_f

    r2 = np.zeros(Z.shape[0], dtype=np.float64)
    valid = n >= min_non_null
    if np.any(valid):
        n_v = n[valid]
        mx_v = sum_x[valid] / n_v
        mz_v = sum_z[valid] / n_v
        var_x = sum_x2[valid] - n_v * mx_v * mx_v
        var_z = sum_z2[valid] - n_v * mz_v * mz_v
        cov = sum_xz[valid] - n_v * mx_v * mz_v
        denom = var_x * var_z
        ok = denom > 0.0
        r = np.zeros_like(cov)
        r[ok] = cov[ok] / np.sqrt(denom[ok])
        r2[valid] = r * r

    return r2, n


def batch_r2_many_vs_many(X, MX, Z, MZ, min_non_null=2, sanitize=False, precomputed_n=None):
    """Compute masked many-to-many r^2 and overlap counts for two tiles."""
    if X.ndim != 2 or MX.ndim != 2 or Z.ndim != 2 or MZ.ndim != 2:
        raise ValueError("X, MX, Z, and MZ must be 2D arrays")
    if X.shape != MX.shape or Z.shape != MZ.shape:
        raise ValueError("value and valid shapes must match for both sides")
    if X.shape[1] != Z.shape[1]:
        raise ValueError("X and Z must have the same sample dimension")

    X = X.astype(np.float64, copy=False)
    Z = Z.astype(np.float64, copy=False)
    MX_bool = MX.astype(bool, copy=False)
    MZ_bool = MZ.astype(bool, copy=False)

    has_nan_x = False
    has_nan_z = False
    if sanitize:
        has_nan_x = np.isnan(X).any()
        has_nan_z = np.isnan(Z).any()
        if has_nan_x:
            MX_bool = MX_bool.copy()
            MX_bool[np.isnan(X)] = False
        if has_nan_z:
            MZ_bool = MZ_bool.copy()
            MZ_bool[np.isnan(Z)] = False

    Xv = np.where(MX_bool, X, 0.0)
    Zv = np.where(MZ_bool, Z, 0.0)
    MXf = MX_bool.astype(np.float64, copy=False)
    MZf = MZ_bool.astype(np.float64, copy=False)

    if precomputed_n is not None and not has_nan_x and not has_nan_z:
        n = np.asarray(precomputed_n, dtype=np.int64)
    else:
        n = (MXf @ MZf.T).astype(np.int64, copy=False)
    sum_x = Xv @ MZf.T
    sum_z = MXf @ Zv.T
    sum_x2 = (Xv * Xv) @ MZf.T
    sum_z2 = MXf @ ((Zv * Zv).T)
    sum_xz = Xv @ Zv.T

    r2 = np.zeros_like(sum_xz, dtype=np.float64)
    valid = n >= min_non_null
    if not np.any(valid):
        return r2, n

    n_f = n.astype(np.float64, copy=False)
    mean_x = np.zeros_like(sum_x, dtype=np.float64)
    mean_z = np.zeros_like(sum_z, dtype=np.float64)
    np.divide(sum_x, n_f, out=mean_x, where=valid)
    np.divide(sum_z, n_f, out=mean_z, where=valid)

    var_x = sum_x2 - n_f * mean_x * mean_x
    var_z = sum_z2 - n_f * mean_z * mean_z
    cov = sum_xz - n_f * mean_x * mean_z
    denom = var_x * var_z
    ok = valid & (denom > 0.0)
    if np.any(ok):
        r = np.zeros_like(sum_xz, dtype=np.float64)
        r[ok] = cov[ok] / np.sqrt(denom[ok])
        r2[ok] = r[ok] * r[ok]
    return r2, n
