"""
Compute angular area RSRs for foreground and background angles.
"""

from typing import Union, Sequence, Tuple, List
import numpy as np
from .helpers import compute_rsr, compute_RMSD


def compute_rsp(
    theta_fgs: Union[np.ndarray, Sequence[np.ndarray]],
    theta_bg: np.ndarray,
    scanning_window: float,
    resolution: int,
    scanning_range: np.ndarray,
    mode: str = "absolute",
    expected_model: str = "local",
    normalize: bool = True,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[List[np.ndarray], List[np.ndarray], np.ndarray],
]:
    """Compute RSP curves from foreground/background angular distributions.

    Args:
        theta_fgs: Either a single np.ndarray or list of np.ndarrays containing
                          foreground angular values (radians) for one or more features.
        theta_bg (np.ndarray): Background angular values (radians).
        scanning_window (float): Width of the angular scanning window (radians).
        resolution (int): Number of bins within the scanning window.
        scanning_range (np.ndarray): List of center angles to scan over.
        mode (str): Either "absolute" or "relative".
        expected_model (str): 'random' for uniform-circle null, 'local' for background-adapted null.
        normalize (bool): If True, normalize the output areas by the background RSR.

    Returns:
        tuple: When single feature provided:
               - In absolute mode: (fg_curve, expected_fg_curve, bg_curve)
               - In relative mode: (fg_curve, bg_curve)
               When multiple features provided:
               - In absolute mode: (fg_curves, exp_fg_curves, bg_curve)
               - In relative mode: (fg_curves, bg_curve)
               where fg_curves and exp_fg_curves are lists of arrays.
    """
    bins = np.linspace(-scanning_window / 2, scanning_window / 2, resolution + 1)
    single_feature = isinstance(theta_fgs, np.ndarray)
    if single_feature:
        theta_fgs = [theta_fgs]
    elif not isinstance(theta_fgs, (list, tuple)):
        raise ValueError(
            "theta_fgs must be a numpy array or a list/tuple of numpy arrays"
        )

    num_features = len(theta_fgs)
    coverages = [len(theta_fg) / len(theta_bg) for theta_fg in theta_fgs]

    fg_curves = [np.zeros(len(scanning_range)) for _ in range(num_features)]
    exp_fg_curves = [np.zeros(len(scanning_range)) for _ in range(num_features)]
    bg_curve = np.zeros(len(scanning_range))

    for i, center in enumerate(scanning_range):
        bg_rsr = compute_rsr(
            theta_bg,
            center,
            scanning_window,
            bins,
        )
        bg_curve[i] = bg_rsr / bg_rsr if normalize else bg_rsr

        for j, theta_fg in enumerate(theta_fgs):
            fg_rsr = compute_rsr(theta_fg, center, scanning_window, bins)
            if expected_model == "random":
                M = int(len(theta_bg) * coverages[j])
                theta_uniform = np.random.default_rng().uniform(-np.pi, np.pi, size=M)
                exp_rsr = compute_rsr(theta_uniform, center, scanning_window, bins)
            else:  # local
                exp_rsr = bg_rsr * coverages[j]

            fg_val = fg_rsr / bg_rsr if normalize else fg_rsr
            exp_val = exp_rsr / bg_rsr if normalize else exp_rsr

            fg_curves[j][i] = fg_val
            exp_fg_curves[j][i] = exp_val

    if mode == "absolute":
        if single_feature:
            return fg_curves[0], exp_fg_curves[0], bg_curve
        else:
            return fg_curves, exp_fg_curves, bg_curve
    elif mode == "relative":
        rel_curves = []
        for fg_curve, exp_curve in zip(fg_curves, exp_fg_curves):
            with np.errstate(divide="ignore", invalid="ignore"):
                rel_curve = (fg_curve + 1e-12) / (exp_curve + 1e-12)
            rel_curves.append(rel_curve)
        if single_feature:
            return rel_curves[0], bg_curve
        else:
            return rel_curves, bg_curve
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'absolute' or 'relative'.")


def compute_RMSD_with_permutation(
    theta_fg: np.ndarray,
    theta_bg: np.ndarray,
    scanning_window: float,
    resolution: int,
    scanning_range: np.ndarray,
    mode: str = "relative",
    n_perm: int = 1000,
    min_cells: int = 10,
) -> dict:
    """
    FG = theta_fg (subset), BG = theta_bg (all cells).
    We compute:
      - bg_ref     : the BG distribution once
      - obs_rmsd   : RMSD(FG_obs, bg_ref)
      - null_rmsds : RMSD(FG_perm, bg_ref) over many perms
    """
    n_fg, n_bg = len(theta_fg), len(theta_bg)
    M = min(n_fg, n_bg)
    if M < min_cells:
        raise ValueError(f"Need ≥{min_cells} cells, got {n_fg} vs {n_bg}")

    # 1) compute the BG reference ONCE
    #    (we ignore FG when building the BG curve)
    #    mode="relative" returns (fg_curve, bg_curve)
    if mode == "absolute": # compare expected to observed
        _, bg_ref, _ = compute_rsp(
            theta_bg,  # pretend FG=BG so we only care about bg out
            theta_bg,
            scanning_window,
            resolution,
            scanning_range,
            mode="absolute",
        )
    else:  # mode="relative"
        _, bg_ref = compute_rsp(
            theta_bg,  # pretend FG=BG so we only care about bg out
            theta_bg,
            scanning_window,
            resolution,
            scanning_range,
            mode="relative",
        )

    # 2) observed RMSD
    if mode == "absolute":
        fg_obs, _, _ = compute_rsp(
            theta_fg,
            theta_bg,
            scanning_window,
            resolution,
            scanning_range,
            mode="absolute",
        )
    else:  # mode="relative"
        fg_obs, _ = compute_rsp(
            theta_fg,
            theta_bg,
            scanning_window,
            resolution,
            scanning_range,
            mode="relative",
        )
    obs_rmsd = compute_RMSD(fg_obs, bg_ref)

    # 3) permutation null
    perm_rmsds = np.empty(n_perm)
    for i in range(n_perm):
        # draw a random subset of BG as a fake FG
        fg_perm = np.random.choice(theta_bg, size=M, replace=False)

        if mode == "absolute":
            fg_p, _, _ = compute_rsp(
                fg_perm,
                theta_bg,
                scanning_window,
                resolution,
                scanning_range,
                mode="absolute",
            )
        else:  # mode="relative"
            fg_p, _ = compute_rsp(
                fg_perm,
                theta_bg,
                scanning_window,
                resolution,
                scanning_range,
                mode="relative",
            )
        perm_rmsds[i] = compute_RMSD(fg_p, bg_ref)

    # 4) summarize null
    null_mean = perm_rmsds.mean()
    null_std = perm_rmsds.std(ddof=1)
    z_score = (obs_rmsd - null_mean) / null_std
    p_val = float((np.sum(perm_rmsds >= obs_rmsd) + 1) / (n_perm + 1))
    ci_low, ci_up = np.percentile(perm_rmsds, [2.5, 97.5])

    return {
        "RMSD": obs_rmsd,
        "p_value": p_val,
        "z_score": z_score,
        "null_mean": null_mean,
        "null_std": null_std,
        "ci_lower": ci_low,
        "ci_upper": ci_up,
        "null_dist": perm_rmsds,
    }
