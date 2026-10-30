#!/usr/bin/env python3
"""Regenerate every data figure and schematic of the paper (Figures 1 and 3-16).

Fig. 2 (the pipeline flowchart) is drawn in Graphviz from
``assets/pipeline_flowchart.dot`` by ``scripts/11_pipeline_flowchart.py``.

All figures are drawn at their printed width (6.5 in for full-width figures), so
the font sizes below are the sizes on the page. They carry no embedded titles (the
caption is the title), use the paper's terms (UAV, TLS, unsmoothed baseline), order
masts bow to stern, and label axes with units in brackets.

Figures 8 and 16 read results/significance_*.json (scripts/15_significance_and_ablations.py).
Output: ``figures/paper/figNN_<name>.png`` (300 dpi; ``--dpi 600`` for print) and
``.pdf`` (vector, with the point clouds rasterized).
Runtime: about two minutes. Requires both ``.mat`` files in ``./data/``.
"""
import importlib
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from _common import base_parser, ensure_dirs, load_drone, load_laser  # noqa: E402
from mastfit.baseline import naive_pipeline  # noqa: E402
from mastfit.crossval import cross_validate  # noqa: E402
from mastfit.doming import doming_decomposition  # noqa: E402
from mastfit.geometry import irls_axis, orthonormal_basis, pca_axis, robust_circle, taubin_circle  # noqa: E402
from mastfit.io import coregister, robust_clean  # noqa: E402
from mastfit.pipeline import fit_mast  # noqa: E402

NAMES = ["Foremast", "Mainmast", "Middle Mast", "Mizzenmast", "Jiggermast"]
FULL = 6.5                      # printed text width [in]
UAV, TLS = "#2a78d6", "#eb6834"  # sensor colours (validated categorical pair)
VERDICT = {"significant": "#1f7a4d", "not resolved": "#b03a2e"}
INK, MUTED = "#1b1b1b", "#6b6b6b"
MIN_LASER_POINTS = 2000

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.2,
    "axes.edgecolor": "#444444", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": "#444444", "ytick.color": "#444444", "grid.color": "#dddddd",
    "grid.linewidth": 0.5, "legend.frameon": False, "savefig.dpi": 300,
    "figure.dpi": 100, "pdf.fonttype": 42,
})


def mast_colour(i):
    """Ordinal bow-to-stern colour (viridis, darkest at the bow)."""
    return plt.cm.viridis(np.linspace(0, 0.8, 5)[i])


def label(i):
    return f"{i + 1} {NAMES[i]}"


def save(fig, outdir, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  wrote {name}.png/.pdf")


# --------------------------------------------------------------------------- Fig. 1
def fig01_estimand(outdir):
    """Schematic: horizontal offsets are exaggerated, heights are to scale. The straight
    reference axis is the least-squares line through the centreline (Section 3.7), so the
    bending profile b(z) takes both signs."""
    H = 50.0
    z = np.linspace(0, H, 300)
    lean = 0.15 * z                          # lean of the mast (rake exaggerated)
    bend = 2.2 * np.sin(np.pi * z / H)       # bending (exaggerated)
    centre = lean + bend
    coef = np.polyfit(z, centre, 1)          # least-squares straight reference axis
    axis = np.polyval(coef, z)
    radius = 0.85 - 0.007 * z
    fig, ax = plt.subplots(figsize=(3.0, 3.8))
    ax.fill_betweenx(z, centre - radius, centre + radius, color="#cfd8e3", alpha=.7, lw=0,
                     label="mast surface (tapered)")
    ax.plot(axis, z, "--", color="#b03a2e", lw=1.2,
            label="fitted straight axis $\\mathbf{c}_0 + z\\,\\mathbf{d}$")
    ax.plot(centre, z, color=INK, lw=1.6, label="as-built centreline $\\mathbf{c}(z)$")
    for zk in (3, 25):
        k = np.argmin(abs(z - zk))
        ax.annotate("", xy=(centre[k], z[k]), xytext=(axis[k], z[k]),
                    arrowprops=dict(arrowstyle="<->", color="#1f7a4d", lw=1, shrinkA=0, shrinkB=0))
        ax.text(max(centre[k], axis[k]) + 0.25, z[k], "$b(z)$", color="#1f7a4d", va="center")
    x0 = coef[1]
    ax.plot([x0, x0], [0, 14], color=MUTED, lw=0.8)
    arc_t = np.linspace(np.pi / 2, np.pi / 2 - np.arctan(coef[0]), 30)
    ax.plot(x0 + 11.5 * np.cos(arc_t), 11.5 * np.sin(arc_t), color="#b03a2e", lw=0.8)
    ax.text(x0 + 0.15, 12.4, "rake", color="#b03a2e", fontsize=7)
    ax.set_xlabel("horizontal offset (exaggerated)")
    ax.set_ylabel("height $z$ [m]")
    ax.set_xticks([])
    ax.set_xlim(-2.5, 10.5)
    ax.set_ylim(0, H + 1)
    ax.legend(loc="upper left", fontsize=6.5)
    ax.grid(alpha=.5, axis="y")
    save(fig, outdir, "fig01_estimand")


# --------------------------------------------------------------------------- Fig. 3
def fig03_trimmed_circle(outdir):
    rg = np.random.default_rng(3)
    R = 0.35
    th = rg.uniform(0, 2 * np.pi, 90)
    r = R + rg.normal(0, 0.008, 90)
    ring = np.column_stack([r * np.cos(th), r * np.sin(th)])
    n = 14
    ta = rg.normal(0, 0.35, n)
    ra = R + rg.uniform(0.08, 0.30, n)
    att = np.column_stack([ra * np.cos(ta), ra * np.sin(ta)])
    P = np.vstack([ring, att])
    c0, r0, _, _ = taubin_circle(P)
    c1, r1, _, _, keep = robust_circle(P)
    fig, ax = plt.subplots(figsize=(3.4, 3.1))
    tt = np.linspace(0, 2 * np.pi, 200)
    ax.plot(c0[0] + r0 * np.cos(tt), c0[1] + r0 * np.sin(tt), ":", color="#b03a2e", lw=1,
            label="circle fitted to all points")
    ax.plot(c1[0] + r1 * np.cos(tt), c1[1] + r1 * np.sin(tt), "--", color=INK, lw=1,
            label="circle after trimmed refit")
    ax.scatter(*ring.T, s=9, color="#9aa6b2", edgecolor="#5b6670", lw=.3, label="mast wall points (retained)", zorder=3)
    ax.scatter(*P[~keep].T, s=12, color=TLS, edgecolor="#7a2f10", lw=.3, label="attachment points (trimmed)", zorder=4)
    ax.scatter(*c0, marker="x", s=40, color="#b03a2e", lw=1.4, label="centre, all points", zorder=5)
    ax.scatter(*c1, marker="o", s=26, color="#1f7a4d", edgecolor="white", lw=.6, label="centre, trimmed refit", zorder=5)
    ax.text(1.04, 0.08, "points with $|\\rho_i-\\tilde\\rho|>2.5\\,$MAD are removed\nand the circle is refitted (up to four passes)",
            transform=ax.transAxes, fontsize=6.5, va="top", color=INK)
    ax.set_aspect("equal")
    ax.set_xlim(-0.46, 0.72)
    ax.set_ylim(-0.46, 0.46)
    ax.set_xlabel("$x$ [m]")
    ax.set_ylabel("$y$ [m]")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=6.5)
    ax.grid(alpha=.5)
    save(fig, outdir, "fig03_trimmed_circle_fit")


# --------------------------------------------------------------------------- Fig. 4
def fig04_input(outdir, dt):
    from matplotlib.ticker import MaxNLocator
    fig = plt.figure(figsize=(FULL, 3.0))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    for i, M in enumerate(dt):
        S = M[:: max(1, len(M) // 5000)]
        ax.scatter(S[:, 0], S[:, 1], S[:, 2], s=0.3, color=mast_colour(i), alpha=.5, rasterized=True)
    for axis in (ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(3))
    ax.set_xticks([])
    ax.set_xlabel("$x$", labelpad=-10)
    ax.set_ylabel("$y$, bow → stern [m]", labelpad=-2)
    ax.set_zlabel("$z$ [m]", labelpad=-4)
    ax.tick_params(pad=-2)
    try:
        ax.set_box_aspect((0.35, 2.0, 1.0))
    except Exception:  # pragma: no cover
        pass
    ax.view_init(elev=14, azim=-58)
    ax.set_title("(a)", loc="left")
    ax2 = fig.add_subplot(1, 2, 2)
    pos = np.array([[np.median(t[:, 0]), np.median(t[:, 1])] for t in dt])
    for i, M in enumerate(dt):
        S = M[:: max(1, len(M) // 4000)]
        ax2.scatter(S[:, 1], S[:, 0], s=0.4, color=mast_colour(i), alpha=.5, rasterized=True)
        ax2.text(pos[i, 1], -1.35, f"{i + 1}", ha="center", va="top", fontsize=7, fontweight="bold")
    order = np.argsort(pos[:, 1])
    for j in range(4):
        a, b = order[j], order[j + 1]
        ax2.annotate("", xy=(pos[b, 1], 1.45), xytext=(pos[a, 1], 1.45),
                     arrowprops=dict(arrowstyle="<->", color=MUTED, lw=.7))
        ax2.text((pos[a, 1] + pos[b, 1]) / 2, 1.55, f"{pos[b, 1] - pos[a, 1]:.1f} m", ha="center", fontsize=6.5)
    ax2.set_ylim(-1.9, 1.9)
    fig.text(0.5, 0.01, "  ·  ".join(label(i) for i in range(5)), ha="center", va="bottom", fontsize=6.5)
    ax2.set_xlabel("$y$, bow → stern [m]")
    ax2.set_ylabel("$x$, transverse [m]")
    ax2.set_title("(b)", loc="left")
    ax2.grid(alpha=.5)
    fig.subplots_adjust(wspace=0.3, bottom=0.2)
    save(fig, outdir, "fig04_input_data")


# --------------------------------------------------------------------------- Fig. 5
def fig05_quality(outdir, res):
    fig, axs = plt.subplots(2, 5, figsize=(FULL, 4.0), sharey="col")
    for i, r in enumerate(res):
        h = r.all_slice_height
        d = r.all_slice_radius_m * 2000
        o = r.all_slice_ovality_pct
        fl, bd = r.flare_removed, r.band_flagged & ~r.flare_removed
        for row, (vals, xl) in enumerate(((d, "diameter [mm]"), (o, "ovality [%]"))):
            ax = axs[row, i]
            ax.plot(vals[~fl], h[~fl], "-", color=mast_colour(i), lw=.8)
            ax.scatter(vals[~fl & ~bd], h[~fl & ~bd], s=4, color=mast_colour(i), zorder=3)
            ax.scatter(vals[fl], h[fl], s=10, color="#9a9a9a", marker="s", zorder=4)
            ax.scatter(vals[bd], h[bd], s=12, color="#b03a2e", marker="D", zorder=4)
            ax.set_xlabel(xl)
            ax.grid(alpha=.5)
            ax.locator_params(axis="x", nbins=3)
            if i == 0:
                ax.set_ylabel("height [m]")
        axs[0, i].set_title(label(i))
    handles = [Line2D([], [], marker="o", ls="-", color=mast_colour(2), ms=3, lw=.8, label="retained slice"),
               Line2D([], [], marker="s", ls="", color="#9a9a9a", ms=4, label="removed by the flare test"),
               Line2D([], [], marker="D", ls="", color="#b03a2e", ms=4, label="flagged as a rigging band")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(hspace=0.45, wspace=0.35, bottom=0.17)
    save(fig, outdir, "fig05_slice_quality_control")


# --------------------------------------------------------------------------- Fig. 6
def fig06_flare(outdir, drone):
    fig, axs = plt.subplots(1, 2, figsize=(4.6, 3.0), sharey=True)
    for ax, i in zip(axs, (2, 3)):
        off, h0, b0 = naive_pipeline(drone[i], quality_control=False, return_profile=True)
        on, h1, b1 = naive_pipeline(drone[i], quality_control=True, return_profile=True)
        ax.plot(b0, h0, color="#b03a2e", lw=1, label=f"off: max {off:.0f} mm")
        ax.plot(b1, h1, color="#1f7a4d", lw=1.2, label=f"on: max {on:.0f} mm")
        ax.set_title(label(i))
        ax.set_xlabel("total lateral bend [mm]")
        ax.grid(alpha=.5)
        ax.legend(loc="center right", fontsize=6.5, title="flare test", title_fontsize=6.5)
    axs[0].set_ylabel("height [m]")
    fig.subplots_adjust(wspace=0.12)
    save(fig, outdir, "fig06_flare_ablation")


# --------------------------------------------------------------------------- Fig. 7
def fig07_profiles(outdir, res):
    fig, axs = plt.subplots(2, 5, figsize=(FULL, 3.6), sharey="col")
    for i, r in enumerate(res):
        s = r.median_sigma_mm
        for row, (b, name) in enumerate(((r.bend_transverse_mm, "transverse"),
                                         (r.bend_longitudinal_mm, "longitudinal"))):
            ax = axs[row, i]
            ax.fill_betweenx(r.height, b - 2 * s, b + 2 * s, color=mast_colour(i), alpha=.18, lw=0)
            ax.plot(b, r.height, color=mast_colour(i), lw=1.3)
            ax.axvline(0, color=MUTED, lw=.6)
            ax.grid(alpha=.5)
            ax.locator_params(axis="x", nbins=4)
            if i == 0:
                ax.set_ylabel(f"{name}\nheight [m]")
            if row == 1:
                ax.set_xlabel("bend [mm]")
        axs[0, i].set_title(f"{label(i)}\nrake {r.rake_deg:.2f}°")
    fig.subplots_adjust(hspace=0.3, wspace=0.35)
    save(fig, outdir, "fig07_bending_profiles")


# --------------------------------------------------------------------------- Fig. 8
def load_null(path="results/significance_B.json"):
    """Null total bends of straight phantoms (scripts/15_significance_and_ablations.py --sections B)."""
    import json
    with open(path) as f:
        return json.load(f)["B_null_samples"]


def fig08_significance(outdir, res, null):
    # Each measured total bend is tested against the null
    # distribution of the total bend from straight synthetic masts of the same
    # geometry under the assumed error model; significant if above its 95th percentile.
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    data = [np.asarray(null[n]) for n in NAMES]
    bp = ax.boxplot(data, positions=np.arange(5), widths=.36, patch_artist=True, showfliers=False,
                    whis=(5, 95), zorder=3)
    for patch in bp["boxes"]:
        patch.set_facecolor("#d5dbe2"); patch.set_edgecolor(MUTED)
    for k in ("medians", "whiskers", "caps"):
        for e in bp[k]:
            e.set_color(MUTED)
    for i, r in enumerate(res):
        b0 = data[i]
        p95 = np.percentile(b0, 95)
        p = (1 + np.sum(b0 >= r.max_bend_mm)) / (len(b0) + 1)
        sig = r.max_bend_mm > p95
        ax.bar(i, r.max_bend_mm, width=.62, color=VERDICT["significant"] if sig else "#9a9a9a",
               alpha=.9, zorder=2)
        ax.plot([i - .36, i + .36], [p95, p95], color=INK, lw=1.2, zorder=4)
        if p <= 1 / (len(b0) + 1) + 1e-12:
            ptxt = "p < 0.005"
        elif p < 0.1:
            ptxt = f"p = {p:.3f}"
        else:
            ptxt = f"p = {p:.2f}"
        note = "\n(UAV only)" if i == 0 else ""
        ax.text(i, max(r.max_bend_mm, b0.max()) + 4, f"{r.max_bend_mm:.0f} mm\n{ptxt}{note}", ha="center",
                va="bottom", fontsize=6.5)
    ax.set_xticks(np.arange(5), [label(i).replace(" ", "\n", 1) for i in range(5)])
    ax.set_ylabel("total bend [mm]")
    ax.set_ylim(0, 215)
    handles = [Patch(color=VERDICT["significant"], label="measured, significant (5%)"),
               Patch(color="#9a9a9a", label="measured, not significant"),
               Patch(facecolor="#d5dbe2", edgecolor=MUTED, label="null: straight masts (box 25–75%, whiskers 5–95%)"),
               Line2D([], [], color=INK, lw=1.2, label="null 95th percentile")]
    ax.legend(handles=handles, loc="upper right", fontsize=6)
    ax.grid(alpha=.5, axis="y")
    save(fig, outdir, "fig08_significance")


# --------------------------------------------------------------------------- Fig. 9
def fig09_doming(outdir, dm):
    fig, axs = plt.subplots(1, 2, figsize=(4.8, 3.4), sharey=True)
    g = dm["grid"]
    for i in range(5):
        axs[0].plot(dm["transverse"][i], g, color=mast_colour(i), lw=1, label=label(i))
        axs[1].plot(dm["corrected_transverse"][i], g, color=mast_colour(i), lw=1)
    axs[0].plot(dm["shared_transverse"], g, "--", color="#b03a2e", lw=1.6,
                label=f"shared ({dm['shared_fraction_transverse'] * 100:.0f}% of variance)")
    axs[0].set_title("(a) transverse profiles", loc="left")
    axs[1].set_title("(b) after removing the shared component", loc="left")
    for ax in axs:
        ax.axvline(0, color=MUTED, lw=.6)
        ax.set_xlabel("transverse bend [mm]")
        ax.grid(alpha=.5)
    axs[0].set_ylabel("normalized height")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=6.5, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(wspace=0.08, bottom=0.27)
    save(fig, outdir, "fig09_shared_curvature")


# --------------------------------------------------------------------------- Fig. 10
def fig10_synthetic(outdir, em, fl, dom):
    fig, axs = plt.subplots(1, 3, figsize=(FULL, 2.9), gridspec_kw={"width_ratios": [1, 1, 1.25]})
    c = fl["C"]
    vals = [c["truth"], c["baseline_off"], c["baseline_on"]]
    axs[0].bar([0, 1, 2], vals, color=["#555555", "#b03a2e", "#1f7a4d"], width=.6)
    for k, v in enumerate(vals):
        axs[0].text(k, v + 3, f"{v:.0f}", ha="center", fontsize=6.5)
    axs[0].set_xticks([0, 1, 2], ["true", "test off", "test on"])
    axs[0].set_ylabel("maximum bend [mm]")
    axs[0].set_title("(a) flare mechanism", loc="left")
    axs[0].set_ylim(0, 145)
    data = [em["white"], em["correlated"]]
    bp = axs[1].boxplot(data, widths=.5, patch_artist=True, showfliers=False, tick_labels=["white", "correlated"])
    for patch, col in zip(bp["boxes"], ("#9aa6b2", "#b03a2e")):
        patch.set_facecolor(col)
        patch.set_alpha(.55)
    for med in bp["medians"]:
        med.set_color(INK)
    for k, dd in enumerate(data, start=1):
        axs[1].scatter(np.full(len(dd), k) + np.linspace(-.12, .12, len(dd)), dd, s=8, color=INK, zorder=3)
        axs[1].text(k, max(dd) + 0.5, f"mean {np.mean(dd):.1f} mm", fontsize=6.5, ha="center", va="bottom")
    axs[1].set_ylabel("bending-profile RMSE [mm]")
    axs[1].set_title("(b) error model", loc="left")
    axs[1].set_xlim(.5, 2.5)
    axs[1].set_ylim(0, max(20.0, 1.25 * max(max(x) for x in data)))
    g = dom["grid"]
    for i in range(5):
        axs[2].plot(dom["recovered"][i], g, color=mast_colour(i), lw=.8, alpha=.8)
    axs[2].plot(dom["shared_injected"], g, color=INK, lw=1.6, label=f"injected shared ({dom['fraction_injected'] * 100:.0f}%)")
    axs[2].plot(dom["shared_recovered"], g, "--", color="#b03a2e", lw=1.4,
                label=f"recovered shared ({dom['fraction_recovered'] * 100:.0f}%)")
    axs[2].set_xlabel("bend [mm]")
    axs[2].set_ylabel("normalized height")
    axs[2].set_title("(c) doming recovery", loc="left")
    axs[2].legend(loc="upper left", bbox_to_anchor=(0.0, -0.22), fontsize=6.5)
    for ax in axs:
        ax.grid(alpha=.5)
    fig.subplots_adjust(wspace=0.42, bottom=0.3)
    save(fig, outdir, "fig10_synthetic_validation")


# --------------------------------------------------------------------------- Fig. 11
def fig11_phantom(outdir, pm):
    panels = [("sigma", "radial residual σ [mm]"), ("trimmed", "points trimmed [%]"),
              ("coverage", "angular coverage [%]"), ("density", "point density [points/m]")]
    cols = ["#555555", "#9aa6b2", "#1f7a4d"]
    fig, axs = plt.subplots(1, 4, figsize=(FULL, 2.2))
    for ax, (key, yl) in zip(axs, panels):
        data = [pm["raw_real"][key], pm["raw_orig"][key], pm["raw_match"][key]]
        bp = ax.boxplot(data, widths=.55, patch_artist=True, showfliers=False, whis=(10, 90),
                        tick_labels=["real", "original", "matched"])
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c)
            patch.set_alpha(.55)
        for med in bp["medians"]:
            med.set_color(INK)
        ax.set_ylabel(yl)
        ax.grid(alpha=.5, axis="y")
        ax.tick_params(axis="x", labelsize=6.5)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    axs[2].set_ylim(70, 102)
    p10 = np.percentile(pm["raw_real"]["coverage"], 10)
    axs[2].text(1.12, p10, f"10th pct. {p10:.0f}%", fontsize=6.5, va="center")
    fig.subplots_adjust(wspace=0.55)
    save(fig, outdir, "fig11_phantom_match")


# --------------------------------------------------------------------------- Fig. 12
def fig12_floor(outdir, res, floors):
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    data = [floors[n] for n in NAMES]
    bp = ax.boxplot(data, widths=.5, patch_artist=True, showfliers=False, whis=(0, 100),
                    tick_labels=[label(i).replace(" ", "\n", 1) for i in range(5)])
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(mast_colour(i))
        patch.set_alpha(.45)
    for med in bp["medians"]:
        med.set_color(INK)
    for i, r in enumerate(res):
        p90 = np.percentile(floors[NAMES[i]], 90)
        ax.plot([i + .7, i + 1.3], [p90, p90], color=INK, lw=1.1)
        ax.scatter(i + 1, r.max_bend_mm, marker="D", s=26, color="#b03a2e", edgecolor="white", lw=.5, zorder=5)
        ax.text(i + 1.18, r.max_bend_mm, f"{r.max_bend_mm:.0f}", va="center", fontsize=6.5)
    handles = [Patch(color="#9aa6b2", alpha=.6, label="floor over 10 phantom realizations"),
               Line2D([], [], color=INK, lw=1.1, label="90th percentile"),
               Line2D([], [], marker="D", ls="", color="#b03a2e", ms=4, label="measured bend")]
    ax.legend(handles=handles, loc="upper right", fontsize=6.5)
    ax.set_ylabel("bend or floor [mm]")
    ax.set_ylim(0, 175)
    ax.grid(alpha=.5, axis="y")
    save(fig, outdir, "fig12_per_mast_floor")


# --------------------------------------------------------------------------- Fig. 13
def _upright(P):
    c0, d0, _ = pca_axis(P)
    c, d = irls_axis(P, c0, d0)
    B, d = orthonormal_basis(d)
    t = (P - c) @ d
    return np.column_stack([(P - c) @ B.T, t - t.min()])


def fig13_sensors(outdir, drone, laser):
    """Side views of each cloud in its own axis frame (horizontal scale exaggerated)."""
    panels = [("UAV", drone[3], "4 Mizzenmast", UAV), ("TLS", laser[3], "4 Mizzenmast", TLS),
              ("TLS", laser[0], "1 Foremast", TLS)]
    fig, axs = plt.subplots(1, 3, figsize=(4.4, 3.6), sharey=True)
    for ax, (sensor, P, name, col) in zip(axs, panels):
        if len(P) >= MIN_LASER_POINTS:
            U = _upright(P)
        else:                                    # too sparse for a robust axis: centre on the median
            U = np.column_stack([P[:, 0] - np.median(P[:, 0]), P[:, 1] - np.median(P[:, 1]),
                                 P[:, 2] - P[:, 2].min()])
        S = U[:: max(1, len(U) // 15000)]
        ax.scatter(S[:, 0], S[:, 2], s=0.4, color=col, alpha=.5, rasterized=True, lw=0)
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(0, 55)
        ax.set_xlabel("offset from axis [m]")
        note = "" if len(P) >= MIN_LASER_POINTS else "\ntoo sparse to fit"
        ax.set_title(f"{sensor}, {name}\n{len(P):,} points{note}", fontsize=7,
                     color="#b03a2e" if note else INK)
        ax.grid(alpha=.5)
    axs[0].set_ylabel("height along the mast axis [m]")
    fig.subplots_adjust(wspace=0.15)
    save(fig, outdir, "fig13_two_sensors")


# --------------------------------------------------------------------------- Fig. 14 / 15
def fig14_crossval_profiles(outdir, cv, usable):
    fig, axs = plt.subplots(1, len(usable), figsize=(FULL, 2.9), sharey=False)
    for ax, i in zip(axs, usable):
        t, lo = cv["transverse"]["per_mast"][i], cv["longitudinal"]["per_mast"][i]
        tu = np.hypot(t["uav"], lo["uav"])
        tt = np.hypot(t["tls"], lo["tls"])
        ax.plot(tu, t["z"], color=UAV, lw=1.3, label="UAV")
        ax.plot(tt, t["z"], color=TLS, lw=1.3, label="TLS")
        ax.set_title(f"{label(i)}\nmax: UAV {tu.max():.0f}, TLS {tt.max():.0f} mm", fontsize=7)
        ax.set_xlabel("total bend [mm]")
        ax.grid(alpha=.5)
        ax.locator_params(axis="x", nbins=4)
    axs[0].set_ylabel("height in the common frame [m]")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(wspace=0.3, bottom=0.25)
    save(fig, outdir, "fig14_uav_vs_tls_profiles")


def fig15_bland_altman(outdir, cv, usable):
    fig, axs = plt.subplots(1, 2, figsize=(FULL, 3.0))
    for ax, comp in zip(axs, ("longitudinal", "transverse")):
        s = cv[comp]
        for i in usable:
            pmast = s["per_mast"][i]
            ax.scatter((pmast["uav"] + pmast["tls"]) / 2, pmast["diff"], s=6, color=mast_colour(i),
                       alpha=.8, label=label(i))
        ax.axhline(0, color=INK, lw=.8)
        for sign in (-1, 1):
            ax.axhline(sign * s["loa_mm"], color="#b03a2e", lw=.9, ls="--")
        ax.text(ax.get_xlim()[1], s["loa_mm"], f" +{s['loa_mm']:.0f} mm", va="center", fontsize=6.5, color="#b03a2e")
        ax.text(ax.get_xlim()[1], -s["loa_mm"], f" −{s['loa_mm']:.0f} mm", va="center", fontsize=6.5, color="#b03a2e")
        ax.set_title(f"{comp}: r = {s['corr']:+.2f}, RMS difference {s['rms_diff_mm']:.0f} mm", loc="left")
        ax.set_xlabel("mean of UAV and TLS bend [mm]")
        ax.set_ylabel("UAV − TLS [mm]")
        ax.grid(alpha=.5)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01), markerscale=1.5)
    fig.subplots_adjust(wspace=0.35, right=0.93, bottom=0.25)
    save(fig, outdir, "fig15_bland_altman")


def fig16_kernel_response(outdir, transfer):
    """Amplitude transfer of the Gaussian smoother (scripts/15_significance_and_ablations.py, section C).
    Lines: theory exp(-2 pi^2 L^2 / lambda^2); markers: full pipeline on synthetic masts."""
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    lam = np.geomspace(3, 100, 400)
    styles = {"2.0": ("#7fa7d9", ":"), "3.0": (INK, "-"), "5.0": ("#b03a2e", "--")}
    handles = []
    for L, (col, ls) in styles.items():
        Lf = float(L)
        ax.plot(lam, np.exp(-2 * np.pi ** 2 * Lf ** 2 / lam ** 2), color=col, ls=ls, lw=1.2)
        pts = transfer[L]
        x = np.array([float(w) for w in pts])
        y = np.array([pts[w]["pipeline"] for w in pts])
        ax.plot(x, y, "o", ms=3.5, color=col, mec="white", mew=.4)
        handles.append(Line2D([], [], color=col, ls=ls, lw=1.2, marker="o", ms=3.5, mec="white", mew=.4,
                              label=f"L = {Lf:g} m"))
    ax.axvspan(7, 8, color="#e8d9b5", alpha=.8, lw=0)
    handles.append(Patch(color="#e8d9b5", label="inter-yard scale (7–8 m)"))
    ax.axhline(0.5, color=MUTED, lw=.6, ls="-")
    ax.set_xscale("log")
    ax.set_xticks([4, 6, 8, 10, 16, 20, 30, 50, 100], ["4", "6", "8", "10", "16", "20", "30", "50", "100"])
    ax.set_xlim(3.5, 100)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("wavelength of the bend $\\lambda$ [m]")
    ax.set_ylabel("amplitude transfer")
    ax.legend(handles=handles, loc="upper left", fontsize=6.5)
    ax.grid(alpha=.5)
    save(fig, outdir, "fig16_kernel_response")


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--dpi", type=int, default=300,
                        help="raster resolution (600 dpi or more for print)")
    args = parser.parse_args()
    plt.rcParams["savefig.dpi"] = args.dpi
    outdir = os.path.join(args.outdir, "paper")
    ensure_dirs(outdir)
    drone, laser = load_drone(args), load_laser(args)
    print("fitting ...")
    res = [fit_mast(P, name=NAMES[i], n_slices=args.slices, bandwidth_m=args.bandwidth)
           for i, P in enumerate(drone)]
    dt, lt = coregister(drone, laser)
    usable = [i for i in range(5) if len(laser[i]) >= MIN_LASER_POINTS]
    rd = [fit_mast(dt[i], n_slices=args.slices, bandwidth_m=args.bandwidth) for i in range(5)]
    rl = [fit_mast(robust_clean(lt[i]), n_slices=args.slices, bandwidth_m=args.bandwidth)
          if i in usable else None for i in range(5)]
    cv = cross_validate(rd, rl, usable)
    print("synthetic studies ...")
    syn = importlib.import_module("05_synthetic_validation")
    em, fl, dom = syn.experiment_error_model(), syn.experiment_flare(), syn.experiment_doming()
    pm = importlib.import_module("08_phantom_match").compute(args)
    floors = importlib.import_module("10_per_mast_accuracy").compute_floors(10)

    print("figures ->", outdir)
    fig01_estimand(outdir)
    fig03_trimmed_circle(outdir)
    fig04_input(outdir, dt)
    fig05_quality(outdir, res)
    fig06_flare(outdir, drone)
    fig07_profiles(outdir, res)
    fig08_significance(outdir, res, load_null())
    fig09_doming(outdir, doming_decomposition(res))
    fig10_synthetic(outdir, em, fl, dom)
    fig11_phantom(outdir, pm)
    fig12_floor(outdir, res, floors)
    if os.path.exists("results/significance_ACD.json"):
        import json
        with open("results/significance_ACD.json") as f:
            fig16_kernel_response(outdir, json.load(f)["C_transfer"])
    fig13_sensors(outdir, drone, laser)
    fig14_crossval_profiles(outdir, cv, usable)
    fig15_bland_altman(outdir, cv, usable)


if __name__ == "__main__":
    main()
