"""Download and subset KPMP snCv3 dataset for TAL analysis.

This example downloads the KPMP single-nucleus RNA-seq dataset using
``biorsp.utils.download_kpmp`` and loads it into an :class:`~anndata.AnnData`
object.  The script verifies the presence of the ``TAL`` (thick ascending
limb) population, ensures a UMAP embedding is available, and splits the
TAL cells into subtypes when ``subclass.l2`` annotations exist.  For a
set of genes of interest we compute foreground/background masks across
quantile bins so that downstream BioRSP analyses can be performed.
"""

from __future__ import annotations

import argparse
import logging
from typing import Dict, Iterable, List

import numpy as np
from anndata import AnnData
from scipy import sparse

import matplotlib.pyplot as plt

import biorsp.io as bio_io
import biorsp.utils as bio_utils
from biorsp.preprocess import Preprocessor
import biorsp.rsp as rsp

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


GENES_OF_INTEREST: List[str] = [
    "UMOD",
    "SLC12A1",
    "CLDN16",
    "CLDN10",
]

# Representative injury markers used in example figures
AKI_GENE = "HAVCR1"  # Up-regulated in acute kidney injury
CKD_GENE = "COL1A1"  # Biased toward chronic kidney disease


def _compute_masks(
    adata: AnnData, genes: Iterable[str], n_bins: int
) -> Dict[str, List[Dict[str, np.ndarray]]]:
    """Return foreground/background masks for each gene across quantile bins."""

    thresholds = np.linspace(0, 1, n_bins + 1)
    results: Dict[str, List[Dict[str, np.ndarray]]] = {}
    for gene in genes:
        if gene not in adata.var_names:
            logger.warning("%s not in var_names; skipping", gene)
            continue

        col = adata[:, gene].X
        expr = col.toarray().ravel() if sparse.issparse(col) else np.ravel(col)
        nz = expr[expr > 0]
        if nz.size == 0:
            logger.warning("%s has no non-zero expression", gene)
            quants = np.zeros_like(thresholds)
        else:
            quants = np.quantile(nz, thresholds)

        gene_masks: List[Dict[str, np.ndarray]] = []
        for i in range(n_bins):
            low, high = quants[i], quants[i + 1]
            fg = (expr > low) & (expr <= high)
            gene_masks.append({"foreground": fg, "background": ~fg})
        results[gene] = gene_masks

    return results


def _quantile_threshold(expr: np.ndarray, q: float) -> float:
    """Helper to compute non-zero expression quantile threshold."""

    nz = expr[expr > 0]
    if nz.size == 0:
        return float("nan")
    return float(np.quantile(nz, q))


def plot_umap_deciles(
    adata: AnnData,
    gene: str,
    cond_key: str = "condition",
    *,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot UMAP foreground/background for a gene across conditions."""

    if gene not in adata.var_names:
        raise KeyError(f"Gene '{gene}' not found in var_names")

    conds = ["Control", "AKI", "CKD", "Pooled"]
    expr = adata[:, gene].X
    expr = expr.toarray().ravel() if sparse.issparse(expr) else np.ravel(expr)
    th = _quantile_threshold(expr, 0.9)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    coords = adata.obsm["X_umap"]
    for ax, cond in zip(axes.flat, conds):
        mask = (
            adata.obs[cond_key] == cond
            if cond != "Pooled" and cond_key in adata.obs
            else np.ones(adata.n_obs, dtype=bool)
        )
        sub = coords[mask]
        vals = expr[mask]
        fg = vals >= th
        ax.scatter(sub[~fg, 0], sub[~fg, 1], c="lightgray", s=4, linewidths=0)
        ax.scatter(sub[fg, 0], sub[fg, 1], c="red", s=4, linewidths=0)
        ax.set_title(cond)
        ax.axis("off")

    fig.suptitle(f"{gene} top 10% foreground")
    return fig


def load_dataset(n_bins: int) -> tuple[AnnData, Dict[str, AnnData], Dict[str, List[Dict[str, np.ndarray]]]]:
    """Download KPMP sn dataset, subset TAL cells and compute masks."""

    logger.info("Downloading KPMP snCv3 dataset…")
    h5ad_path = bio_utils.download_kpmp(variant="sn")
    logger.info("Dataset downloaded to %s", h5ad_path)

    logger.info("Loading dataset into AnnData…")
    adata = bio_io.load_data(h5ad_path)
    logger.info("Data shape: %s", adata.shape)

    label_col = None
    for c in ("subclass.l1", "cluster", "nephron_segment"):
        if c in adata.obs and "TAL" in adata.obs[c].unique():
            label_col = c
            break
    if label_col is None:
        raise ValueError("TAL population not annotated in obs")

    tal_cells = adata[adata.obs[label_col] == "TAL"].copy()
    logger.info("TAL subset shape: %s", tal_cells.shape)

    if "X_umap" not in tal_cells.obsm:
        logger.info("Computing UMAP embedding")
        Preprocessor().run(
            tal_cells,
            qc=False,
            normalize=False,
            reduction={"method": "UMAP", "n_components": 2},
        )

    subtypes: Dict[str, AnnData] = {}
    if "subclass.l2" in tal_cells.obs:
        for subtype in tal_cells.obs["subclass.l2"].unique():
            st = tal_cells[tal_cells.obs["subclass.l2"] == subtype].copy()
            subtypes[subtype] = st
            logger.info("Subtype %s shape: %s", subtype, st.shape)
    else:
        logger.warning("subclass.l2 column not found; skipping subtype split")

    masks = _compute_masks(tal_cells, GENES_OF_INTEREST, n_bins)

    return tal_cells, subtypes, masks


def make_figures(tal_cells: AnnData, output_prefix: str = "fig") -> None:
    """Generate demonstration figures for the BioRSP poster."""

    # Figure 1: decile-based UMAPs for two marker genes
    for gene in (AKI_GENE, CKD_GENE):
        fig = plot_umap_deciles(tal_cells, gene)
        fig.savefig(f"{output_prefix}_umap_{gene}.png", dpi=300)
        plt.close(fig)

    # Placeholder stubs for additional figures. These can be expanded with
    # real analyses once the notebook is further developed.
    logger.info("[TODO] Implement radar plots, profile curves, heatmaps, \n"
                "co-regulation networks, and benchmark comparisons.")


def main(argv: List[str] | None = None):
    parser = argparse.ArgumentParser(description="Prepare KPMP TAL data")
    parser.add_argument("--bins", type=int, default=10, help="number of bins for thresholds")
    args = parser.parse_args(argv)

    tal_cells, subtypes, masks = load_dataset(args.bins)
    make_figures(tal_cells)
    return tal_cells, subtypes, masks


if __name__ == "__main__":
    main()
