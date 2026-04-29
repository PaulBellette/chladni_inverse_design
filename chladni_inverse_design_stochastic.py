#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy",
#   "scipy",
#   "pillow",
#   "matplotlib",
#   "torch",
# ]
# ///
#!/usr/bin/env python3
"""
chladni_inverse_design.py

Toy inverse design for time-averaged Chladni-like intensity fields on a
simply-supported square plate.

The model is:

    E(x,y) = sum_k b_k sin^2(m_k*pi*x) sin^2(n_k*pi*y)

with b_k >= 0.

This version supports basin-shaped objectives:

1. Convert binary mask to a distance-transform target energy.
2. Ignore a boundary margin in the loss, because the simply-supported basis
   forces E=0 at the boundary.
3. Optimise either:
   - weighted MSE against a smooth basin target, or
   - soft basin classification/ranking loss using PyTorch.

Example:

    uv run chladni_inverse_design.py mask.png \
        --w-max 80 \
        --size 128 \
        --blur 1.0 \
        --boundary-margin 8 \
        --loss weighted-mse \
        --preview result.png

PyTorch version:

    uv run chladni_inverse_design.py mask.png \
        --w-max 120 \
        --size 128 \
        --boundary-margin 8 \
        --loss soft-basin \
        --steps 4000 \
        --lr 0.03 \
        --ranking-weight 0.2 \
        --preview result.png
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter, label as ndi_label
from scipy.optimize import lsq_linear


@dataclass
class Mode:
    m: int
    n: int
    omega_nd: float


def load_binary_mask(path: Path, size: int, threshold: float = 0.5) -> np.ndarray:
    """
    Load image as grayscale, resize to size x size, return boolean mask.

    True means desired low-energy basin / sand-collecting region.
    """
    img = Image.open(path).convert("L")
    img = img.resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float64) / 255.0
    return arr > threshold


def make_boundary_valid_mask(size: int, margin: int) -> np.ndarray:
    """
    Valid pixels for loss evaluation.

    Simply-supported basis forces E=0 on the boundary, so by default we ignore
    a strip around the edge.
    """
    valid = np.ones((size, size), dtype=bool)
    if margin > 0:
        valid[:margin, :] = False
        valid[-margin:, :] = False
        valid[:, :margin] = False
        valid[:, -margin:] = False
    return valid


def make_distance_basin_target(
    mask: np.ndarray,
    sigma: float,
    blur: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a basin-shaped target energy.

    mask=True is desired low energy.

    distance is zero inside/at the target mask and increases away from it.

    E_target = 1 - exp(-d^2 / sigma^2)

    So:
        target region -> 0
        far away      -> approaches 1
    """
    if mask.sum() == 0:
        raise ValueError("Mask has no positive pixels. Need a non-empty target basin.")

    outside_distance = distance_transform_edt(~mask).astype(np.float64)

    sigma = max(float(sigma), 1e-6)
    target = 1.0 - np.exp(-(outside_distance**2) / (sigma**2))

    if blur > 0:
        target = gaussian_filter(target, sigma=blur)

    target -= target.min()
    denom = target.max() - target.min()
    if denom > 1e-12:
        target /= denom

    return target, outside_distance


def make_loss_weights(
    distance: np.ndarray,
    valley_weight: float,
    valley_sigma: float,
    valid: np.ndarray,
) -> np.ndarray:
    """
    Higher weights near the desired basin.

        w = 1 + valley_weight * exp(-d^2 / valley_sigma^2)

    Invalid boundary pixels get zero weight.
    """
    valley_sigma = max(float(valley_sigma), 1e-6)
    weights = 1.0 + valley_weight * np.exp(-(distance**2) / (valley_sigma**2))
    weights = weights.astype(np.float64)
    weights[~valid] = 0.0
    return weights


def build_modes(w_max: float) -> list[Mode]:
    """
    Select simply-supported square plate modes satisfying:

        omega_nd = m^2 + n^2 <= w_max

    Physical frequency is proportional to omega_nd.
    """
    modes: list[Mode] = []
    max_index = int(np.floor(np.sqrt(w_max)))

    for m in range(1, max_index + 1):
        for n in range(1, max_index + 1):
            omega_nd = m * m + n * n
            if omega_nd <= w_max:
                modes.append(Mode(m=m, n=n, omega_nd=float(omega_nd)))

    modes.sort(key=lambda mode: (mode.omega_nd, mode.m, mode.n))
    return modes


def build_design_matrix(size: int, modes: list[Mode]) -> np.ndarray:
    """
    Build A where each column is:

        psi_mn(x,y) = sin^2(m*pi*x) sin^2(n*pi*y)
    """
    x = np.linspace(0.0, 1.0, size, endpoint=True)
    y = np.linspace(0.0, 1.0, size, endpoint=True)
    X, Y = np.meshgrid(x, y, indexing="xy")

    cols = []
    for mode in modes:
        phi = np.sin(mode.m * np.pi * X) * np.sin(mode.n * np.pi * Y)
        cols.append((phi * phi).reshape(-1))

    return np.stack(cols, axis=1)


def solve_weighted_mse(
    A: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    ridge: float,
) -> np.ndarray:
    """
    Solve:

        min_b sum_i w_i (A_i b - target_i)^2 + ridge ||b||^2
        s.t. b >= 0

    using scipy lsq_linear.
    """
    y = target.reshape(-1)
    w = weights.reshape(-1)

    keep = w > 0
    A_use = A[keep]
    y_use = y[keep]
    sqrt_w = np.sqrt(w[keep])

    A_w = A_use * sqrt_w[:, None]
    y_w = y_use * sqrt_w

    if ridge > 0:
        n_modes = A.shape[1]
        A_aug = np.vstack([A_w, np.sqrt(ridge) * np.eye(n_modes)])
        y_aug = np.concatenate([y_w, np.zeros(n_modes)])
    else:
        A_aug = A_w
        y_aug = y_w

    result = lsq_linear(
        A_aug,
        y_aug,
        bounds=(0.0, np.inf),
        method="trf",
        lsmr_tol="auto",
        verbose=0,
    )

    if not result.success:
        print(f"Warning: solver did not fully converge: {result.message}")

    return result.x

def build_component_index_lists(binary: np.ndarray) -> list[np.ndarray]:
    """
    Return flattened pixel-index arrays, one per connected component.

    Used to sample valley positives in a component-balanced way, so small
    components like eyes/mouth are not drowned out by a large outline.
    """
    labelled, n_components = ndi_label(binary.astype(bool))
    components: list[np.ndarray] = []

    for component_id in range(1, n_components + 1):
        idx = np.flatnonzero((labelled == component_id).reshape(-1))
        if len(idx) > 0:
            components.append(idx)

    return components


def sample_component_balanced_indices(
    rng: np.random.Generator,
    components: list[np.ndarray],
    n_samples: int,
) -> np.ndarray:
    """
    Sample approximately uniformly over connected components, then uniformly
    over pixels within each selected component.
    """
    if not components:
        raise ValueError("Cannot sample from zero connected components.")

    component_ids = rng.integers(0, len(components), size=n_samples)
    out = np.empty(n_samples, dtype=np.int64)

    for i, component_id in enumerate(component_ids):
        component = components[int(component_id)]
        out[i] = rng.choice(component)

    return out


def solve_soft_basin_torch(
    A: np.ndarray,
    valley_band: np.ndarray,
    offcurve_band: np.ndarray,
    valid: np.ndarray,
    steps: int,
    lr: float,
    ridge: float,
    l1: float,
    temperature: float,
    ranking_weight: float,
    ranking_margin: float,
    ranking_pairs: int,
    seed: int,
    component_balanced: bool,
    hard_negative_fraction: float,
    bce_batch_size: int,
    init_noise: float,
    hard_negative_refresh: int,
) -> np.ndarray:
    """
    Optimise a basin classification objective.

    Energy is nonnegative:
        b = softplus(raw_b)

    Basin probability is:
        p(mask=True | x) = sigmoid((tau - E(x)) / temperature)

    tau is learned.

    BCE encourages low energy inside the target mask and high energy outside.

    Optional ranking loss samples inside/outside pairs and encourages:

        E_inside + margin < E_outside
    """
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise ImportError("soft-basin loss requires PyTorch: pip install torch") from exc

    rng = np.random.default_rng(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)

    A_t = torch.tensor(A, dtype=torch.float32, device=device)

    label = np.zeros_like(valid, dtype=np.float32)
    label[valley_band] = 1.0

    supervised = valley_band | offcurve_band

    label_flat = torch.tensor(label.reshape(-1), device=device)

    valley_idx_np = np.flatnonzero(valley_band.reshape(-1))
    offcurve_idx_np = np.flatnonzero(offcurve_band.reshape(-1))
    supervised_idx_np = np.flatnonzero(supervised.reshape(-1))

    if len(valley_idx_np) == 0:
        raise ValueError("No valley-band pixels remain after applying boundary margin.")
    if len(offcurve_idx_np) == 0:
        raise ValueError("No off-curve pixels remain after applying boundary margin.")

    valley_components = build_component_index_lists(valley_band)
    print(f"Valley connected components: {len(valley_components)}")

    offcurve_idx_t = torch.tensor(offcurve_idx_np, dtype=torch.long, device=device)

    n_modes = A.shape[1]
    raw_b = torch.nn.Parameter(
        -5.0 + init_noise * torch.randn(n_modes, device=device)
    )
    raw_tau = torch.nn.Parameter(torch.tensor(0.2, device=device))

    opt = torch.optim.Adam([raw_b, raw_tau], lr=lr)

    cached_hard_pool: np.ndarray | None = None
    hard_negative_refresh = max(1, int(hard_negative_refresh))

    for step in range(steps):
        opt.zero_grad()

        b = F.softplus(raw_b)
        tau = F.softplus(raw_tau)

        E = A_t @ b
        logits = (tau - E) / max(temperature, 1e-6)

        # Stochastic, class-balanced BCE. This breaks exact symmetry in the
        # per-step gradient and prevents the huge off-curve class from dominating.
        if bce_batch_size > 0:
            n_pos = max(1, bce_batch_size // 2)
            n_neg = max(1, bce_batch_size - n_pos)

            if component_balanced:
                pos_np = sample_component_balanced_indices(
                    rng,
                    valley_components,
                    n_pos,
                )
            else:
                pos_np = rng.choice(
                    valley_idx_np,
                    size=n_pos,
                    replace=True,
                )

            neg_np = rng.choice(
                offcurve_idx_np,
                size=n_neg,
                replace=True,
            )

            batch_np = np.concatenate([pos_np, neg_np])
            rng.shuffle(batch_np)
        else:
            # Full supervised BCE. Mostly useful as an ablation.
            batch_np = supervised_idx_np

        batch_t = torch.tensor(batch_np, dtype=torch.long, device=device)

        bce = F.binary_cross_entropy_with_logits(
            logits[batch_t],
            label_flat[batch_t],
            reduction="mean",
        )

        reg = ridge * torch.mean(b * b) + l1 * torch.mean(b)

        loss = bce + reg

        if ranking_weight > 0 and ranking_pairs > 0:
            if component_balanced:
                valley_sample = sample_component_balanced_indices(
                    rng,
                    valley_components,
                    ranking_pairs,
                )
            else:
                valley_sample = rng.choice(
                    valley_idx_np,
                    size=ranking_pairs,
                    replace=True,
                )

            valley_t = torch.tensor(valley_sample, dtype=torch.long, device=device)

            hard_negative_fraction_clamped = float(
                np.clip(hard_negative_fraction, 0.0, 1.0)
            )
            n_hard = int(round(ranking_pairs * hard_negative_fraction_clamped))
            n_random = ranking_pairs - n_hard

            offcurve_samples = []

            if n_random > 0:
                offcurve_samples.append(
                    rng.choice(
                        offcurve_idx_np,
                        size=n_random,
                        replace=True,
                    )
                )

            if n_hard > 0:
                # Hard negatives are the lowest-energy off-curve pixels, because
                # those are the off-curve regions currently most mistaken for valleys.
                # Refreshing this pool every N steps is much cheaper than a top-k
                # scan on every optimisation step.
                if cached_hard_pool is None or step % hard_negative_refresh == 0:
                    with torch.no_grad():
                        offcurve_E = E[offcurve_idx_t]
                        hard_k = min(max(n_hard * 4, n_hard), offcurve_E.numel())
                        hard_local = torch.topk(-offcurve_E, k=hard_k).indices
                        cached_hard_pool = (
                            offcurve_idx_t[hard_local].detach().cpu().numpy()
                        )

                offcurve_samples.append(
                    rng.choice(
                        cached_hard_pool,
                        size=n_hard,
                        replace=True,
                    )
                )

            offcurve_sample = np.concatenate(offcurve_samples)
            rng.shuffle(offcurve_sample)

            offcurve_t = torch.tensor(offcurve_sample, dtype=torch.long, device=device)

            e_valley = E[valley_t]
            e_offcurve = E[offcurve_t]

            rank_loss = F.relu(e_valley - e_offcurve + ranking_margin).mean()
            loss = loss + ranking_weight * rank_loss

        loss.backward()
        opt.step()

        if step % max(1, steps // 10) == 0 or step == steps - 1:
            print(
                f"step {step:5d} | loss {loss.item():.6g} | "
                f"bce {bce.item():.6g} | tau {tau.item():.4g} | "
                f"mean_b {b.mean().item():.4g}"
            )

    with torch.no_grad():
        coeffs = F.softplus(raw_b).detach().cpu().numpy()

    return coeffs


def reconstruct(A: np.ndarray, coeffs: np.ndarray, size: int) -> np.ndarray:
    """
    Raw physical-ish energy, not normalised.
    """
    return (A @ coeffs).reshape(size, size)


def normalize_image(x: np.ndarray) -> np.ndarray:
    y = x.astype(np.float64).copy()
    y -= y.min()
    denom = y.max() - y.min()
    if denom > 1e-12:
        y /= denom
    return y

def make_curve_distance(mask: np.ndarray) -> np.ndarray:
    """
    Distance in pixels to the target curve/stroke.

    mask=True is the desired low-energy curve.
    """
    if mask.sum() == 0:
        raise ValueError("Mask has no positive pixels. Need a non-empty target curve.")
    return distance_transform_edt(~mask).astype(np.float64)


def make_curve_trench_target(
    curve_distance: np.ndarray,
    valley_sigma: float,
) -> np.ndarray:
    """
    Target energy for a thin curve/trench.

    Low on the curve, high away from the curve:

        E_target = 1 - exp(-d^2 / valley_sigma^2)
    """
    valley_sigma = max(float(valley_sigma), 1e-6)
    target = 1.0 - np.exp(-(curve_distance**2) / (valley_sigma**2))
    target -= target.min()
    denom = target.max() - target.min()
    if denom > 1e-12:
        target /= denom
    return target


def make_curve_loss_weights(
    curve_distance: np.ndarray,
    valid: np.ndarray,
    valley_radius: float,
    offcurve_radius: float,
    valley_weight: float,
    offcurve_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a banded curve objective.

    valley_band:
        pixels near the drawn stroke. These should be low energy.

    offcurve_band:
        pixels sufficiently far from the stroke. These should be high energy.

    ambiguous_band:
        pixels between valley_radius and offcurve_radius. These are ignored.

    weights:
        per-pixel fitting weights for weighted MSE.
    """
    valley_band = (curve_distance <= valley_radius) & valid
    offcurve_band = (curve_distance >= offcurve_radius) & valid
    ambiguous_band = (~valley_band) & (~offcurve_band) & valid

    weights = np.zeros_like(curve_distance, dtype=np.float64)
    weights[valley_band] = valley_weight
    weights[offcurve_band] = offcurve_weight

    return weights, valley_band, offcurve_band, ambiguous_band

def save_preview(
    path: Path,
    mask: np.ndarray,
    valid: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    pred: np.ndarray,
    valley_band: np.ndarray,
    offcurve_band: np.ndarray,
    ambiguous_band: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    pred_norm = normalize_image(pred)
    residual = pred_norm - target

    bands = np.zeros_like(target, dtype=np.float64)
    bands[offcurve_band] = 0.35
    bands[ambiguous_band] = 0.65
    bands[valley_band] = 1.0
    bands[~valid] = 0.0

    fig, axes = plt.subplots(1, 7, figsize=(23, 4))

    axes[0].imshow(mask, cmap="gray")
    axes[0].set_title("Input mask\n target curve")

    axes[1].imshow(valid, cmap="gray")
    axes[1].set_title("Valid loss region")

    axes[2].imshow(bands, cmap="gray")
    axes[2].set_title("Bands\n valley / ignore / off")

    axes[3].imshow(target, cmap="gray")
    axes[3].set_title("Curve trench\n target energy")

    axes[4].imshow(weights, cmap="gray")
    axes[4].set_title("Loss weights")

    axes[5].imshow(pred_norm, cmap="gray")
    axes[5].set_title("Predicted energy\n normalised")

    axes[6].imshow(residual, cmap="coolwarm")
    axes[6].set_title("Residual\n pred - target")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_energy_image(path: Path, energy: np.ndarray) -> None:
    img = (255 * normalize_image(energy)).clip(0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("mask", type=Path, help="Binary mask image. White = desired basin.")
    parser.add_argument("--w-max", type=float, required=True)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument(
        "--loss",
        choices=["weighted-mse", "soft-basin"],
        default="weighted-mse",
    )

    parser.add_argument(
        "--basin-sigma",
        type=float,
        default=10.0,
        help="Pixel scale for distance-transform basin target.",
    )
    parser.add_argument(
        "--blur",
        type=float,
        default=1.0,
        help="Optional blur applied to target energy.",
    )
    parser.add_argument(
        "--boundary-margin",
        type=int,
        default=8,
        help="Pixels ignored near image boundary.",
    )
    parser.add_argument(
        "--valley-weight",
        type=float,
        default=8.0,
        help="Extra loss weight near the desired basin.",
    )
    parser.add_argument(
        "--valley-sigma",
        type=float,
        default=12.0,
        help="Pixel scale for valley weighting.",
    )

    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--l1", type=float, default=0.0)

    # PyTorch soft-basin options
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--temperature", type=float, default=0.08)
    parser.add_argument("--ranking-weight", type=float, default=0.0)
    parser.add_argument("--ranking-margin", type=float, default=0.1)
    parser.add_argument("--ranking-pairs", type=int, default=4096)
    parser.add_argument(
        "--no-component-balanced",
        action="store_true",
        help="Disable component-balanced valley sampling for torch ranking loss.",
    )
    parser.add_argument(
        "--hard-negative-fraction",
        type=float,
        default=0.5,
        help=(
            "Fraction of off-curve ranking samples drawn from the currently "
            "lowest-energy off-curve pixels. 0 disables hard negatives."
        ),
    )
    parser.add_argument(
        "--hard-negative-refresh",
        type=int,
        default=50,
        help="Refresh hard-negative pool every N optimisation steps.",
    )
    parser.add_argument(
        "--bce-batch-size",
        type=int,
        default=4096,
        help=(
            "Number of pixels sampled for stochastic balanced BCE. "
            "Use 0 for full supervised BCE."
        ),
    )
    parser.add_argument(
        "--init-noise",
        type=float,
        default=0.05,
        help="Std-dev of random noise added to initial modal logits.",
    )
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--out", type=Path, default=Path("chladni_solution.npz"))
    parser.add_argument("--preview", type=Path, default=Path("chladni_preview.png"))
    parser.add_argument(
        "--energy-image",
        type=Path,
        default=Path("chladni_energy.png"),
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--curve-target-sigma",
        type=float,
        default=4.0,
        help="Pixel scale for curve-trench target energy. Smaller means thinner trench.",
    )
    parser.add_argument(
        "--valley-radius",
        type=float,
        default=2.5,
        help="Pixels from mask treated as desired low-energy valley.",
    )
    parser.add_argument(
        "--offcurve-radius",
        type=float,
        default=10.0,
        help="Pixels from mask treated as definitely off-curve / should be higher energy.",
    )
    parser.add_argument(
        "--valley-fit-weight",
        type=float,
        default=10.0,
        help="Weighted-MSE importance for curve valley pixels.",
    )
    parser.add_argument(
        "--offcurve-fit-weight",
        type=float,
        default=1.0,
        help="Weighted-MSE importance for off-curve pixels.",
    )
    args = parser.parse_args()

    mask = load_binary_mask(args.mask, size=args.size, threshold=args.threshold)
    valid = make_boundary_valid_mask(args.size, args.boundary_margin)

    curve_distance = make_curve_distance(mask)

    target = make_curve_trench_target(
        curve_distance=curve_distance,
        valley_sigma=args.curve_target_sigma,
    )

    if args.blur > 0:
        target = gaussian_filter(target, sigma=args.blur)
        target = normalize_image(target)

    weights, valley_band, offcurve_band, ambiguous_band = make_curve_loss_weights(
        curve_distance=curve_distance,
        valid=valid,
        valley_radius=args.valley_radius,
        offcurve_radius=args.offcurve_radius,
        valley_weight=args.valley_fit_weight,
        offcurve_weight=args.offcurve_fit_weight,
    )

    modes = build_modes(args.w_max)
    if not modes:
        raise ValueError("No modes selected. Increase --w-max.")

    print(f"Selected {len(modes)} modes up to nondimensional w_max={args.w_max}")
    print(
        f"Highest selected mode: m={modes[-1].m}, "
        f"n={modes[-1].n}, omega_nd={modes[-1].omega_nd}"
    )
    print(f"Loss: {args.loss}")

    A = build_design_matrix(args.size, modes)

    if args.loss == "weighted-mse":
        coeffs = solve_weighted_mse(
            A=A,
            target=target,
            weights=weights,
            ridge=args.ridge,
        )
    elif args.loss == "soft-basin":
        coeffs = solve_soft_basin_torch(
            A=A,
            valley_band=valley_band,
            offcurve_band=offcurve_band,
            valid=valid,
            steps=args.steps,
            lr=args.lr,
            ridge=args.ridge,
            l1=args.l1,
            temperature=args.temperature,
            ranking_weight=args.ranking_weight,
            ranking_margin=args.ranking_margin,
            ranking_pairs=args.ranking_pairs,
            seed=args.seed,
            component_balanced=not args.no_component_balanced,
            hard_negative_fraction=args.hard_negative_fraction,
            bce_batch_size=args.bce_batch_size,
            init_noise=args.init_noise,
            hard_negative_refresh=args.hard_negative_refresh,
        )
    else:
        raise ValueError(f"Unknown loss: {args.loss}")

    pred = reconstruct(A, coeffs, args.size)
    pred_norm = normalize_image(pred)

    valid_float = valid.astype(np.float64)
    mse = np.sum(valid_float * (pred_norm - target) ** 2) / max(valid_float.sum(), 1)
    inside_mean = float(pred_norm[mask & valid].mean()) if np.any(mask & valid) else np.nan
    outside_mean = float(pred_norm[(~mask) & valid].mean()) if np.any((~mask) & valid) else np.nan
    valley_mean = float(pred_norm[valley_band].mean()) if np.any(valley_band) else np.nan
    offcurve_mean = float(pred_norm[offcurve_band].mean()) if np.any(offcurve_band) else np.nan
    ambiguous_mean = float(pred_norm[ambiguous_band].mean()) if np.any(ambiguous_band) else np.nan

    print(f"Mean energy valley band:     {valley_mean:.6g}")
    print(f"Mean energy off-curve band:  {offcurve_mean:.6g}")
    print(f"Mean energy ambiguous band:  {ambiguous_mean:.6g}")
    print(f"Offcurve - valley margin:    {offcurve_mean - valley_mean:.6g}")
    print(f"Valid-region normalised MSE: {mse:.6g}")
    print(f"Mean energy inside mask:     {inside_mean:.6g}")
    print(f"Mean energy outside mask:    {outside_mean:.6g}")
    print(f"Outside - inside margin:     {outside_mean - inside_mean:.6g}")

    mode_table = np.array(
        [(mode.m, mode.n, mode.omega_nd, coeffs[i]) for i, mode in enumerate(modes)],
        dtype=[
            ("m", "i4"),
            ("n", "i4"),
            ("omega_nd", "f8"),
            ("coefficient", "f8"),
        ],
    )

    np.savez_compressed(
        args.out,
        mask=mask.astype(np.uint8),
        valid=valid.astype(np.uint8),
        target=target,
        weights=weights,
        energy=pred,
        energy_normalized=pred_norm,
        coefficients=coeffs,
        modes_m=np.array([mode.m for mode in modes]),
        modes_n=np.array([mode.n for mode in modes]),
        modes_omega_nd=np.array([mode.omega_nd for mode in modes]),
        mode_table=mode_table,
        w_max=args.w_max,
        size=args.size,
        loss=args.loss,
        basin_sigma=args.basin_sigma,
        blur=args.blur,
        boundary_margin=args.boundary_margin,
        valley_weight=args.valley_weight,
        valley_sigma=args.valley_sigma,
        ridge=args.ridge,
        l1=args.l1,
        curve_distance=curve_distance,
        valley_band=valley_band.astype(np.uint8),
        offcurve_band=offcurve_band.astype(np.uint8),
        ambiguous_band=ambiguous_band.astype(np.uint8),
        curve_target_sigma=args.curve_target_sigma,
        valley_radius=args.valley_radius,
        offcurve_radius=args.offcurve_radius,
        valley_fit_weight=args.valley_fit_weight,
        offcurve_fit_weight=args.offcurve_fit_weight,
        component_balanced=not args.no_component_balanced,
        hard_negative_fraction=args.hard_negative_fraction,
        hard_negative_refresh=args.hard_negative_refresh,
        bce_batch_size=args.bce_batch_size,
        init_noise=args.init_noise,
    )

    save_preview(
        args.preview,
        mask,
        valid,
        target,
        weights,
        pred,
        valley_band,
        offcurve_band,
        ambiguous_band,
    )
    save_energy_image(args.energy_image, pred)

    print(f"Saved solution to {args.out}")
    print(f"Saved preview to {args.preview}")
    print(f"Saved energy image to {args.energy_image}")

    ranked = np.argsort(coeffs)[::-1]

    print()
    print(f"Top {min(args.top_k, len(ranked))} modes:")
    for idx in ranked[: args.top_k]:
        if coeffs[idx] <= 1e-12:
            continue
        mode = modes[idx]
        print(
            f"  m={mode.m:2d}, n={mode.n:2d}, "
            f"omega_nd={mode.omega_nd:6.1f}, "
            f"b={coeffs[idx]:.6g}"
        )


if __name__ == "__main__":
    main()