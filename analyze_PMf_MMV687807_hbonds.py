# -*- coding: utf-8 -*-

"""
Hydrogen-bond analysis between MMV687807 (LIG) and the PMf fungal membrane.

Calculates:

1. Number of LIG <-> PMf H-bonds per frame
2. Overall H-bond occupancy
3. Occupancy of each donor <-> acceptor H-bond pair
4. Occupancy by individual membrane residue
5. Occupancy by membrane lipid/residue type
6. Mean H-bond distance and angle
7. Raw H-bond event table

Definition:
    occupancy (%) =
        frames containing the H-bond / sampled frames * 100

H-bond criterion:
    Donor-Acceptor distance <= 3.5 Å
    D-H...A angle >= 150 degrees

Topology:
    PMf_MMV687807_nooverlap.psf

Trajectory:
    PMf_MMV687807_staged.dcd

The PSF is preferred because it contains bond information needed for
reliable donor-hydrogen assignment.
"""

import os
import numpy as np
import pandas as pd

import MDAnalysis as mda
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import (
    HydrogenBondAnalysis
)

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = (
    r"D:\fcm-uerj\artigos-e-producao"
    r"\2027_candida-MMV687807-675968_daniel"
    r"\PMf_MMV687807_SMD"
)

WORKDIR = os.path.join(
    BASE_DIR,
    "NAMD_PMf_MMV687807_staged"
)

TOPOLOGY = os.path.join(
    BASE_DIR,
    "PMf_MMV687807_nooverlap.psf"
)

REFERENCE_PDB = os.path.join(
    BASE_DIR,
    "PMf_MMV687807_nooverlap.pdb"
)

TRAJECTORY = os.path.join(
    WORKDIR,
    "PMf_MMV687807_staged.dcd"
)


# ============================================================
# OUTPUT SUBFOLDER
# ============================================================

OUTDIR = os.path.join(
    WORKDIR,
    "HBONDS_LIG_PMF"
)

os.makedirs(
    OUTDIR,
    exist_ok=True
)


# ============================================================
# OUTPUT FILES
# ============================================================

EVENTS_OUT = os.path.join(
    OUTDIR,
    "hbonds_LIG_PMF_all_events.csv"
)

PER_FRAME_OUT = os.path.join(
    OUTDIR,
    "hbonds_LIG_PMF_per_frame.csv"
)

PAIR_OCC_OUT = os.path.join(
    OUTDIR,
    "hbonds_LIG_PMF_pair_occupancy.csv"
)

RESIDUE_OCC_OUT = os.path.join(
    OUTDIR,
    "hbonds_LIG_PMF_residue_occupancy.csv"
)

LIPID_OCC_OUT = os.path.join(
    OUTDIR,
    "hbonds_LIG_PMF_lipid_type_occupancy.csv"
)

SUMMARY_OUT = os.path.join(
    OUTDIR,
    "hbonds_LIG_PMF_summary.txt"
)

COUNT_PNG = os.path.join(
    OUTDIR,
    "hbonds_LIG_PMF_per_frame.png"
)

PAIR_OCC_PNG = os.path.join(
    OUTDIR,
    "hbonds_LIG_PMF_top10_pair_occupancy.png"
)

RESIDUE_OCC_PNG = os.path.join(
    OUTDIR,
    "hbonds_LIG_PMF_top10_residue_occupancy.png"
)


# ============================================================
# SELECTIONS
# ============================================================

# Ligand
LIG_SEL = "resname LIG"


# ------------------------------------------------------------
# PMf membrane components found in your topology
#
# IMPORTANT:
# CER160 is the PSF topology residue name.
# ------------------------------------------------------------

MEMBRANE_RESNAMES = [
    "CER160",
    "BMAN",
    "DYPC",
    "ERG",
    "POPE",
    "POPI",
    "POPS",
    "PYPE",
    "YOPA",
    "YOPC",
    "YOPE",
]


MEM_SEL = (
    "resname "
    + " ".join(MEMBRANE_RESNAMES)
)


# Combined selection used only to guess H-bond atoms
LIG_MEM_SEL = (
    f"({LIG_SEL}) or ({MEM_SEL})"
)


# ============================================================
# H-BOND SETTINGS
# ============================================================

HBOND_DIST_ANGSTROM = 3.5

HBOND_ANGLE_DEG = 150.0


# Analyze every frame.
#
# Change to 5 or 10 if you deliberately want temporal
# subsampling.
STEP = 1


# Number of H-bonds/residues shown in occupancy figures
TOP_N = 10


# ============================================================
# OPTIONAL TIME SETTINGS
# ============================================================

# Normally MDAnalysis reads trajectory timing automatically.
#
# Leave as None unless you know that the DCD timing metadata
# are incorrect.
#
# Example:
# FRAME_DT_PS = 10.0
#
# would mean one DCD frame every 10 ps.

FRAME_DT_PS = None


# Optional shift of trajectory time
TIME_OFFSET_NS = 0.0


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def atom_label(atom):
    """
    Human-readable atom label.
    """

    return (
        f"{atom.segid}:"
        f"{atom.resname}:"
        f"{atom.resid}:"
        f"{atom.name}"
    )


def residue_label(atom):
    """
    Human-readable residue label.
    """

    return (
        f"{atom.segid}:"
        f"{atom.resname}:"
        f"{atom.resid}"
    )


def calculate_times(u, step):
    """
    Get sampled frame numbers and times in ns.
    """

    frames = []
    times_ns = []

    for ts in u.trajectory[::step]:

        frames.append(
            int(ts.frame)
        )

        if FRAME_DT_PS is not None:

            time_ps = (
                ts.frame
                *
                float(FRAME_DT_PS)
            )

        else:

            try:

                time_ps = float(
                    ts.time
                )

            except Exception:

                time_ps = float(
                    ts.frame
                )

        times_ns.append(
            time_ps / 1000.0
            + TIME_OFFSET_NS
        )

    return (
        frames,
        times_ns
    )


def empty_outputs(
    frames,
    times_ns
):
    """
    Create outputs even if zero H-bonds are found.
    """

    per_frame = pd.DataFrame({
        "frame": frames,
        "time_ns": times_ns,
        "hbonds_count": 0,
        "has_hbond": 0
    })

    per_frame.to_csv(
        PER_FRAME_OUT,
        index=False
    )

    pd.DataFrame().to_csv(
        EVENTS_OUT,
        index=False
    )

    pd.DataFrame().to_csv(
        PAIR_OCC_OUT,
        index=False
    )

    pd.DataFrame().to_csv(
        RESIDUE_OCC_OUT,
        index=False
    )

    pd.DataFrame().to_csv(
        LIPID_OCC_OUT,
        index=False
    )

    with open(
        SUMMARY_OUT,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "No LIG-PMf hydrogen bonds detected.\n"
        )

    save_count_plot(
        per_frame
    )


def save_count_plot(
    per_frame
):
    """
    Number of H-bonds vs trajectory time.
    10 x 10 cm, 300 dpi.
    """

    cm_to_in = 1.0 / 2.54

    plt.figure(
        figsize=(
            10.0 * cm_to_in,
            10.0 * cm_to_in
        ),
        dpi=300
    )

    plt.plot(
        per_frame["time_ns"],
        per_frame["hbonds_count"],
        linewidth=1.0
    )

    plt.xlabel(
        "Time (ns)"
    )

    plt.ylabel(
        "H-bonds (LIG ↔ PMf)"
    )

    plt.title(
        "Ligand–membrane hydrogen bonds"
    )

    plt.grid(
        True,
        linewidth=0.5,
        alpha=0.5
    )

    plt.tight_layout()

    plt.savefig(
        COUNT_PNG,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def save_top_pair_plot(
    occupancy
):
    """
    Top-N donor-acceptor H-bond occupancies.
    """

    if occupancy.empty:
        return

    top = occupancy.head(
        TOP_N
    ).copy()

    labels = (
        top["donor_short"]
        + " → "
        + top["acceptor_short"]
    )

    values = (
        top["occupancy_percent"]
        .to_numpy()
    )

    cm_to_in = 1.0 / 2.54

    plt.figure(
        figsize=(
            13.0 * cm_to_in,
            10.0 * cm_to_in
        ),
        dpi=300
    )

    y = np.arange(
        len(top)
    )

    plt.barh(
        y,
        values
    )

    plt.yticks(
        y,
        labels
    )

    plt.xlabel(
        "Occupancy (%)"
    )

    plt.title(
        "Top LIG–PMf H-bond occupancies"
    )

    plt.xlim(
        0,
        max(
            100,
            values.max() * 1.10
        )
    )

    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.savefig(
        PAIR_OCC_PNG,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def save_top_residue_plot(
    residue_occ
):
    """
    Top-N PMf residues interacting with LIG by H-bond.
    """

    if residue_occ.empty:
        return

    top = residue_occ.head(
        TOP_N
    ).copy()

    cm_to_in = 1.0 / 2.54

    plt.figure(
        figsize=(
            13.0 * cm_to_in,
            10.0 * cm_to_in
        ),
        dpi=300
    )

    y = np.arange(
        len(top)
    )

    plt.barh(
        y,
        top["occupancy_percent"]
    )

    plt.yticks(
        y,
        top["membrane_residue"]
    )

    plt.xlabel(
        "Occupancy (%)"
    )

    plt.title(
        "Top PMf residues forming H-bonds with LIG"
    )

    plt.xlim(
        0,
        max(
            100,
            top["occupancy_percent"].max()
            * 1.10
        )
    )

    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.savefig(
        RESIDUE_OCC_PNG,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 78
    )

    print(
        "LIG <-> PMf HYDROGEN-BOND ANALYSIS"
    )

    print(
        "=" * 78
    )


    # ========================================================
    # CHECK INPUT FILES
    # ========================================================

    if not os.path.isfile(
        TOPOLOGY
    ):

        raise FileNotFoundError(
            f"PSF not found:\n{TOPOLOGY}"
        )


    if not os.path.isfile(
        TRAJECTORY
    ):

        raise FileNotFoundError(
            f"DCD not found:\n{TRAJECTORY}"
        )


    print(
        "\n[INFO] Topology:"
    )

    print(
        TOPOLOGY
    )

    print(
        "\n[INFO] Trajectory:"
    )

    print(
        TRAJECTORY
    )

    print(
        "\n[INFO] Output directory:"
    )

    print(
        OUTDIR
    )


    # ========================================================
    # LOAD SYSTEM
    # ========================================================

    print(
        "\n[INFO] Loading trajectory..."
    )

    u = mda.Universe(
        TOPOLOGY,
        TRAJECTORY
    )


    print(
        f"[INFO] Atoms: "
        f"{u.atoms.n_atoms:,}"
    )

    print(
        f"[INFO] Frames: "
        f"{u.trajectory.n_frames:,}"
    )


    try:

        print(
            f"[INFO] DCD dt: "
            f"{u.trajectory.dt:.6f} ps/frame"
        )

    except Exception:

        pass


    # ========================================================
    # CHECK SELECTIONS
    # ========================================================

    ligand = u.select_atoms(
        LIG_SEL
    )

    membrane = u.select_atoms(
        MEM_SEL
    )


    print(
        f"\n[INFO] Ligand atoms: "
        f"{ligand.n_atoms:,}"
    )

    print(
        f"[INFO] PMf membrane atoms: "
        f"{membrane.n_atoms:,}"
    )


    if ligand.n_atoms == 0:

        raise RuntimeError(
            "No atoms found for resname LIG."
        )


    if membrane.n_atoms == 0:

        raise RuntimeError(
            "No membrane atoms found."
        )


    print(
        "\n[INFO] Membrane residue names detected:"
    )

    detected_resnames = sorted(
        set(
            membrane.resnames
        )
    )

    print(
        ", ".join(
            detected_resnames
        )
    )


    # ========================================================
    # GET FRAME TIMES
    # ========================================================

    frames_sampled, times_ns = (
        calculate_times(
            u,
            STEP
        )
    )

    sampled_frames = len(
        frames_sampled
    )


    print(
        f"\n[INFO] STEP: {STEP}"
    )

    print(
        f"[INFO] Frames analyzed: "
        f"{sampled_frames:,}"
    )


    # ========================================================
    # BUILD H-BOND ANALYSIS
    # ========================================================

    print(
        "\n[INFO] Preparing H-bond selections..."
    )


    hba = HydrogenBondAnalysis(

        universe=u,

        between=[
            LIG_SEL,
            MEM_SEL
        ],

        d_a_cutoff=
            HBOND_DIST_ANGSTROM,

        d_h_a_angle_cutoff=
            HBOND_ANGLE_DEG,

        update_selections=False
    )


    # --------------------------------------------------------
    # Guess only H-bond-capable atoms belonging to
    # ligand + membrane.
    #
    # The PSF supplies charges, masses and bonds.
    # --------------------------------------------------------

    hydrogen_sel = (
        hba.guess_hydrogens(
            LIG_MEM_SEL
        )
    )

    acceptor_sel = (
        hba.guess_acceptors(
            LIG_MEM_SEL
        )
    )


    # Fallbacks in case charge-based guessing returns nothing
    if not hydrogen_sel:

        hydrogen_sel = (
            f"({LIG_MEM_SEL}) "
            "and name H*"
        )


    if not acceptor_sel:

        acceptor_sel = (
            f"({LIG_MEM_SEL}) "
            "and (name O* N*)"
        )


    hba.hydrogens_sel = (
        hydrogen_sel
    )

    hba.acceptors_sel = (
        acceptor_sel
    )


    try:

        n_h = u.select_atoms(
            hydrogen_sel
        ).n_atoms

        n_a = u.select_atoms(
            acceptor_sel
        ).n_atoms

        print(
            f"[INFO] Candidate H atoms: "
            f"{n_h:,}"
        )

        print(
            f"[INFO] Candidate acceptors: "
            f"{n_a:,}"
        )

    except Exception:

        pass


    # ========================================================
    # RUN H-BOND ANALYSIS
    # ========================================================

    print(
        "\n[INFO] Running H-bond analysis..."
    )

    print(
        f"[INFO] D-A cutoff: "
        f"{HBOND_DIST_ANGSTROM:.2f} Å"
    )

    print(
        f"[INFO] D-H-A angle cutoff: "
        f"{HBOND_ANGLE_DEG:.1f}°"
    )


    hba.run(
        step=STEP
    )


    hbonds = (
        hba.results.hbonds
    )


    # ========================================================
    # NO H-BONDS
    # ========================================================

    if (
        hbonds is None
        or len(hbonds) == 0
    ):

        print(
            "\n[INFO] No LIG-PMf "
            "H-bonds detected."
        )

        empty_outputs(
            frames_sampled,
            times_ns
        )

        return


    # ========================================================
    # CREATE EVENT TABLE
    # ========================================================

    events = pd.DataFrame(

        hbonds[:, :6],

        columns=[
            "frame",
            "donor_idx",
            "hydrogen_idx",
            "acceptor_idx",
            "distance_A",
            "angle_deg"
        ]
    )


    # Convert index columns to integers
    for col in [
        "frame",
        "donor_idx",
        "hydrogen_idx",
        "acceptor_idx"
    ]:

        events[col] = (
            events[col]
            .astype(int)
        )


    # ========================================================
    # EXTRA SAFETY:
    # retain strictly LIG <-> PMf events
    # ========================================================

    lig_indices = set(
        ligand.indices.tolist()
    )

    mem_indices = set(
        membrane.indices.tolist()
    )


    mask = [

        (
            row.donor_idx
            in lig_indices
            and
            row.acceptor_idx
            in mem_indices
        )

        or

        (
            row.donor_idx
            in mem_indices
            and
            row.acceptor_idx
            in lig_indices
        )

        for row in events.itertuples()
    ]


    events = (
        events.loc[mask]
        .copy()
        .reset_index(drop=True)
    )


    if events.empty:

        print(
            "\n[INFO] H-bonds existed in the raw analysis, "
            "but no strict LIG-PMf events remained."
        )

        empty_outputs(
            frames_sampled,
            times_ns
        )

        return


    # ========================================================
    # TIME FOR EACH EVENT
    # ========================================================

    frame_time = dict(
        zip(
            frames_sampled,
            times_ns
        )
    )


    events["time_ns"] = (
        events["frame"]
        .map(frame_time)
    )


    # ========================================================
    # HUMAN-READABLE ATOM LABELS
    # ========================================================

    events[
        "donor_atom"
    ] = [

        atom_label(
            u.atoms[i]
        )

        for i in events[
            "donor_idx"
        ]
    ]


    events[
        "hydrogen_atom"
    ] = [

        atom_label(
            u.atoms[i]
        )

        for i in events[
            "hydrogen_idx"
        ]
    ]


    events[
        "acceptor_atom"
    ] = [

        atom_label(
            u.atoms[i]
        )

        for i in events[
            "acceptor_idx"
        ]
    ]


    # ========================================================
    # DIRECTION + MEMBRANE RESIDUE
    # ========================================================

    directions = []

    membrane_residues = []

    membrane_resnames = []


    for row in events.itertuples():

        donor = u.atoms[
            row.donor_idx
        ]

        acceptor = u.atoms[
            row.acceptor_idx
        ]


        if donor.resname == "LIG":

            directions.append(
                "LIG donor -> PMf acceptor"
            )

            membrane_atom = acceptor

        else:

            directions.append(
                "PMf donor -> LIG acceptor"
            )

            membrane_atom = donor


        membrane_residues.append(
            residue_label(
                membrane_atom
            )
        )

        membrane_resnames.append(
            membrane_atom.resname
        )


    events[
        "direction"
    ] = directions

    events[
        "membrane_residue"
    ] = membrane_residues

    events[
        "membrane_resname"
    ] = membrane_resnames


    # ========================================================
    # SAVE RAW EVENTS
    # ========================================================

    events.to_csv(
        EVENTS_OUT,
        index=False
    )


    print(
        f"\n[OK] Raw H-bond events:\n"
        f"{EVENTS_OUT}"
    )


    # ========================================================
    # NUMBER OF H-BONDS PER FRAME
    #
    # Unique donor-H-acceptor triplet per frame
    # ========================================================

    unique_triplets = (
        events[
            [
                "frame",
                "donor_idx",
                "hydrogen_idx",
                "acceptor_idx"
            ]
        ]
        .drop_duplicates()
    )


    counts_by_frame = (
        unique_triplets
        .groupby(
            "frame"
        )
        .size()
    )


    per_frame = pd.DataFrame({

        "frame":
            frames_sampled,

        "time_ns":
            times_ns

    })


    per_frame[
        "hbonds_count"
    ] = (

        per_frame[
            "frame"
        ]

        .map(
            counts_by_frame
        )

        .fillna(0)

        .astype(int)
    )


    per_frame[
        "has_hbond"
    ] = (

        per_frame[
            "hbonds_count"
        ]
        > 0

    ).astype(int)


    per_frame.to_csv(
        PER_FRAME_OUT,
        index=False
    )


    save_count_plot(
        per_frame
    )


    # ========================================================
    # OVERALL OCCUPANCY
    #
    # Percentage of sampled frames with at least
    # one LIG <-> PMf H-bond
    # ========================================================

    overall_occupancy = (

        100.0
        *
        per_frame[
            "has_hbond"
        ].mean()

    )


    mean_hbonds = (
        per_frame[
            "hbonds_count"
        ].mean()
    )

    sd_hbonds = (
        per_frame[
            "hbonds_count"
        ].std()
    )

    median_hbonds = (
        per_frame[
            "hbonds_count"
        ].median()
    )

    max_hbonds = (
        per_frame[
            "hbonds_count"
        ].max()
    )


    # ========================================================
    # PAIR OCCUPANCY
    #
    # Same logic as your previous script:
    #
    # unique donor <-> acceptor heavy-atom pair
    #
    # occupancy =
    # unique frames containing pair / sampled frames
    # ========================================================

    pair_events = (

        events

        .drop_duplicates(
            subset=[
                "frame",
                "donor_idx",
                "acceptor_idx"
            ]
        )
    )


    pair_group = (
        pair_events.groupby(
            [
                "donor_idx",
                "acceptor_idx"
            ]
        )
    )


    pair_occ = (
        pair_group
        .agg(

            donor_atom=(
                "donor_atom",
                "first"
            ),

            acceptor_atom=(
                "acceptor_atom",
                "first"
            ),

            direction=(
                "direction",
                "first"
            ),

            membrane_residue=(
                "membrane_residue",
                "first"
            ),

            membrane_resname=(
                "membrane_resname",
                "first"
            ),

            frames_with_hbond=(
                "frame",
                "nunique"
            ),

            distance_mean_A=(
                "distance_A",
                "mean"
            ),

            distance_sd_A=(
                "distance_A",
                "std"
            ),

            angle_mean_deg=(
                "angle_deg",
                "mean"
            ),

            angle_sd_deg=(
                "angle_deg",
                "std"
            )

        )

        .reset_index()
    )


    pair_occ[
        "sampled_frames"
    ] = sampled_frames


    pair_occ[
        "occupancy_fraction"
    ] = (

        pair_occ[
            "frames_with_hbond"
        ]
        /
        sampled_frames
    )


    pair_occ[
        "occupancy_percent"
    ] = (

        100.0
        *
        pair_occ[
            "occupancy_fraction"
        ]
    )


    # Short labels for figures
    pair_occ[
        "donor_short"
    ] = (

        pair_occ[
            "donor_atom"
        ]
        .apply(
            lambda x:
            ":".join(
                x.split(":")[1:]
            )
        )
    )


    pair_occ[
        "acceptor_short"
    ] = (

        pair_occ[
            "acceptor_atom"
        ]
        .apply(
            lambda x:
            ":".join(
                x.split(":")[1:]
            )
        )
    )


    pair_occ = (

        pair_occ

        .sort_values(
            [
                "occupancy_percent",
                "distance_mean_A"
            ],
            ascending=[
                False,
                True
            ]
        )

        .reset_index(
            drop=True
        )
    )


    pair_occ.to_csv(
        PAIR_OCC_OUT,
        index=False
    )


    save_top_pair_plot(
        pair_occ
    )


    # ========================================================
    # OCCUPANCY BY INDIVIDUAL MEMBRANE RESIDUE
    #
    # At least one ligand H-bond with that residue
    # during a given frame.
    # ========================================================

    residue_frame = (

        events[
            [
                "frame",
                "membrane_residue",
                "membrane_resname"
            ]
        ]

        .drop_duplicates()
    )


    residue_occ = (

        residue_frame

        .groupby(
            [
                "membrane_residue",
                "membrane_resname"
            ]
        )

        .agg(
            frames_with_hbond=(
                "frame",
                "nunique"
            )
        )

        .reset_index()
    )


    residue_occ[
        "sampled_frames"
    ] = sampled_frames


    residue_occ[
        "occupancy_fraction"
    ] = (

        residue_occ[
            "frames_with_hbond"
        ]
        /
        sampled_frames
    )


    residue_occ[
        "occupancy_percent"
    ] = (

        100.0
        *
        residue_occ[
            "occupancy_fraction"
        ]
    )


    residue_occ = (

        residue_occ

        .sort_values(
            "occupancy_percent",
            ascending=False
        )

        .reset_index(
            drop=True
        )
    )


    residue_occ.to_csv(
        RESIDUE_OCC_OUT,
        index=False
    )


    save_top_residue_plot(
        residue_occ
    )


    # ========================================================
    # OCCUPANCY BY LIPID / RESIDUE TYPE
    #
    # Example:
    # POPI, POPS, ERG, CER160, etc.
    #
    # The occupancy here means:
    # percentage of frames with >=1 H-bond to ANY
    # residue belonging to that membrane component.
    # ========================================================

    lipid_frame = (

        events[
            [
                "frame",
                "membrane_resname"
            ]
        ]

        .drop_duplicates()
    )


    lipid_occ = (

        lipid_frame

        .groupby(
            "membrane_resname"
        )

        .agg(
            frames_with_hbond=(
                "frame",
                "nunique"
            )
        )

        .reset_index()
    )


    lipid_occ[
        "sampled_frames"
    ] = sampled_frames


    lipid_occ[
        "occupancy_fraction"
    ] = (

        lipid_occ[
            "frames_with_hbond"
        ]
        /
        sampled_frames
    )


    lipid_occ[
        "occupancy_percent"
    ] = (

        100.0
        *
        lipid_occ[
            "occupancy_fraction"
        ]
    )


    lipid_occ = (

        lipid_occ

        .sort_values(
            "occupancy_percent",
            ascending=False
        )

        .reset_index(
            drop=True
        )
    )


    lipid_occ.to_csv(
        LIPID_OCC_OUT,
        index=False
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    unique_pairs = len(
        pair_occ
    )


    total_events = len(
        events
    )


    with open(
        SUMMARY_OUT,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "LIG <-> PMf HYDROGEN-BOND ANALYSIS\n"
        )

        f.write(
            "=" * 60
            + "\n\n"
        )

        f.write(
            f"Topology: {TOPOLOGY}\n"
        )

        f.write(
            f"Trajectory: {TRAJECTORY}\n\n"
        )

        f.write(
            f"D-A cutoff: "
            f"{HBOND_DIST_ANGSTROM:.2f} Å\n"
        )

        f.write(
            f"D-H-A angle cutoff: "
            f"{HBOND_ANGLE_DEG:.1f} degrees\n"
        )

        f.write(
            f"STEP: {STEP}\n\n"
        )

        f.write(
            f"Total trajectory frames: "
            f"{u.trajectory.n_frames}\n"
        )

        f.write(
            f"Sampled frames: "
            f"{sampled_frames}\n"
        )

        f.write(
            f"Total detected H-bond events: "
            f"{total_events}\n"
        )

        f.write(
            f"Unique donor-acceptor pairs: "
            f"{unique_pairs}\n\n"
        )

        f.write(
            "H-BONDS PER FRAME\n"
        )

        f.write(
            "-" * 40
            + "\n"
        )

        f.write(
            f"Mean ± SD: "
            f"{mean_hbonds:.4f} "
            f"± {sd_hbonds:.4f}\n"
        )

        f.write(
            f"Median: "
            f"{median_hbonds:.4f}\n"
        )

        f.write(
            f"Maximum: "
            f"{max_hbonds}\n\n"
        )

        f.write(
            "OVERALL OCCUPANCY\n"
        )

        f.write(
            "-" * 40
            + "\n"
        )

        f.write(
            "Frames containing at least one "
            "LIG-PMf H-bond:\n"
        )

        f.write(
            f"{overall_occupancy:.3f}%\n\n"
        )

        f.write(
            "TOP H-BOND PAIRS\n"
        )

        f.write(
            "-" * 40
            + "\n"
        )

        top_pairs = (
            pair_occ
            .head(TOP_N)
        )

        for _, row in top_pairs.iterrows():

            f.write(
                f"{row['donor_atom']} -> "
                f"{row['acceptor_atom']}\n"
            )

            f.write(
                f"  Direction: "
                f"{row['direction']}\n"
            )

            f.write(
                f"  Occupancy: "
                f"{row['occupancy_percent']:.3f}%\n"
            )

            f.write(
                f"  Distance: "
                f"{row['distance_mean_A']:.3f} Å\n"
            )

            f.write(
                f"  Angle: "
                f"{row['angle_mean_deg']:.3f}°\n\n"
            )


    # ========================================================
    # CONSOLE RESULTS
    # ========================================================

    print(
        "\n"
        + "=" * 78
    )

    print(
        "RESULTS"
    )

    print(
        "=" * 78
    )


    print(
        f"\nMean H-bonds/frame: "
        f"{mean_hbonds:.4f} "
        f"± {sd_hbonds:.4f}"
    )

    print(
        f"Median H-bonds/frame: "
        f"{median_hbonds:.4f}"
    )

    print(
        f"Maximum H-bonds/frame: "
        f"{max_hbonds}"
    )

    print(
        f"\nOverall H-bond occupancy: "
        f"{overall_occupancy:.3f}%"
    )

    print(
        f"\nUnique donor-acceptor pairs: "
        f"{unique_pairs}"
    )


    print(
        "\nTop 10 H-bond pairs:"
    )


    display_columns = [
        "donor_atom",
        "acceptor_atom",
        "direction",
        "occupancy_percent",
        "distance_mean_A",
        "angle_mean_deg"
    ]


    print(
        pair_occ[
            display_columns
        ]
        .head(10)
        .to_string(
            index=False
        )
    )


    print(
        "\nTop membrane residues:"
    )


    print(
        residue_occ
        .head(10)
        .to_string(
            index=False
        )
    )


    print(
        "\nOccupancy by membrane component:"
    )


    print(
        lipid_occ
        .to_string(
            index=False
        )
    )


    # ========================================================
    # FINAL
    # ========================================================

    print(
        "\n"
        + "=" * 78
    )

    print(
        "ANALYSIS COMPLETE"
    )

    print(
        "=" * 78
    )


    print(
        f"\nAll outputs saved in:\n"
        f"{OUTDIR}"
    )


    print(
        "\nFiles:"
    )

    print(
        "  hbonds_LIG_PMF_all_events.csv"
    )

    print(
        "  hbonds_LIG_PMF_per_frame.csv"
    )

    print(
        "  hbonds_LIG_PMF_pair_occupancy.csv"
    )

    print(
        "  hbonds_LIG_PMF_residue_occupancy.csv"
    )

    print(
        "  hbonds_LIG_PMF_lipid_type_occupancy.csv"
    )

    print(
        "  hbonds_LIG_PMF_summary.txt"
    )

    print(
        "  hbonds_LIG_PMF_per_frame.png"
    )

    print(
        "  hbonds_LIG_PMF_top10_pair_occupancy.png"
    )

    print(
        "  hbonds_LIG_PMF_top10_residue_occupancy.png"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()