# -*- coding: utf-8 -*-

"""
Hydrophobic-contact analysis between MMV675968 (LIG) and the PMf membrane
during the 60 Å / 1.5 ns SMD trajectory.

Operational definition used here:
    Hydrophobic contact =
        ligand nonpolar C/S atom and membrane nonpolar C/S atom
        separated by <= 4.5 Å.

Outputs:
1. Hydrophobic contacts per frame
2. Fraction/percentage of frames with >=1 hydrophobic contact
3. Atom-pair contact occupancy
4. Occupancy by individual membrane residue
5. Occupancy by membrane lipid/residue type
6. Mean/minimum contact distances
7. Raw contact-event table

Topology:
    PMf_LIG_nooverlap.psf

Trajectory:
    PMf_MMV675968_SMD_60A_1p5ns.dcd
"""

import os
import numpy as np
import pandas as pd

import MDAnalysis as mda
from MDAnalysis.lib.distances import capped_distance

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = (
    r"D:\fcm-uerj\artigos-e-producao"
    r"\2027_candida-MMV687807-675968_daniel"
    r"\PMf_MMV675968_SMD"
)

WORKDIR = os.path.join(
    BASE_DIR,
    "NAMD_SMD_MMV675968_60A-1p5ns"
)

TOPOLOGY = os.path.join(
    BASE_DIR,
    "PMf_LIG_nooverlap.psf"
)

TRAJECTORY = os.path.join(
    WORKDIR,
    "PMf_MMV675968_SMD_60A_1p5ns.dcd"
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTDIR = os.path.join(
    WORKDIR,
    "HYDROPHOBIC_CONTACTS_LIG_PMF_SMD"
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
    "hydrophobic_contacts_LIG_PMF_SMD_all_events.csv"
)

PER_FRAME_OUT = os.path.join(
    OUTDIR,
    "hydrophobic_contacts_LIG_PMF_SMD_per_frame.csv"
)

PAIR_OCC_OUT = os.path.join(
    OUTDIR,
    "hydrophobic_contacts_LIG_PMF_SMD_pair_occupancy.csv"
)

RESIDUE_OCC_OUT = os.path.join(
    OUTDIR,
    "hydrophobic_contacts_LIG_PMF_SMD_residue_occupancy.csv"
)

LIPID_OCC_OUT = os.path.join(
    OUTDIR,
    "hydrophobic_contacts_LIG_PMF_SMD_lipid_type_occupancy.csv"
)

SUMMARY_OUT = os.path.join(
    OUTDIR,
    "hydrophobic_contacts_LIG_PMF_SMD_summary.txt"
)

COUNT_PNG = os.path.join(
    OUTDIR,
    "hydrophobic_contacts_LIG_PMF_SMD_per_frame.png"
)

PAIR_OCC_PNG = os.path.join(
    OUTDIR,
    "hydrophobic_contacts_LIG_PMF_SMD_top10_pair_occupancy.png"
)

RESIDUE_OCC_PNG = os.path.join(
    OUTDIR,
    "hydrophobic_contacts_LIG_PMF_SMD_top10_residue_occupancy.png"
)

LIPID_OCC_PNG = os.path.join(
    OUTDIR,
    "hydrophobic_contacts_LIG_PMF_SMD_lipid_type_occupancy.png"
)


# ============================================================
# ANALYSIS SETTINGS
# ============================================================

# Common cutoff for nonpolar atom contacts
CONTACT_CUTOFF_A = 4.5

# Analyze every trajectory frame
STEP = 1

TOP_N = 10


# ============================================================
# SYSTEM SELECTIONS
# ============================================================

LIG_SEL = "resname LIG"


# PMf membrane components from your system
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


# ============================================================
# HYDROPHOBIC ATOM CLASSIFICATION
# ============================================================

def is_hydrophobic_atom(atom):
    """
    Operational nonpolar-heavy-atom definition.

    Includes carbon and sulfur atoms.

    Additionally uses PSF partial charge, when available, to
    remove strongly polar/charged C/S atoms.

    abs(charge) <= 0.5 is retained.

    This primarily captures:
    - lipid hydrocarbon chains
    - sterol carbon skeleton
    - ligand carbon-rich hydrophobic regions
    """

    name = atom.name.upper()

    # Ignore hydrogens
    if name.startswith("H"):
        return False

    # Determine likely element from atom name
    likely_carbon = name.startswith("C")
    likely_sulfur = name.startswith("S")

    if not (
        likely_carbon
        or likely_sulfur
    ):
        return False

    # PSF charges are available in this system
    try:

        charge = float(
            atom.charge
        )

        if abs(charge) > 0.5:
            return False

    except Exception:
        pass

    return True


# ============================================================
# LABEL FUNCTIONS
# ============================================================

def atom_label(atom):

    return (
        f"{atom.segid}:"
        f"{atom.resname}:"
        f"{atom.resid}:"
        f"{atom.name}"
    )


def residue_label(atom):

    return (
        f"{atom.segid}:"
        f"{atom.resname}:"
        f"{atom.resid}"
    )


# ============================================================
# PLOT: CONTACTS PER FRAME
# ============================================================

def save_per_frame_plot(df):

    cm_to_in = 1.0 / 2.54

    plt.figure(
        figsize=(
            10.0 * cm_to_in,
            10.0 * cm_to_in
        ),
        dpi=300
    )

    plt.plot(
        df["time_ns"],
        df["hydrophobic_contacts"],
        linewidth=1.0
    )

    plt.xlabel(
        "Time (ns)"
    )

    plt.ylabel(
        "Hydrophobic contacts"
    )

    plt.title(
        "LIG–PMf hydrophobic contacts during SMD"
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


# ============================================================
# PLOT: TOP PAIR OCCUPANCY
# ============================================================

def save_pair_plot(df):

    if df.empty:
        return

    top = df.head(
        TOP_N
    ).copy()

    labels = (
        top["ligand_atom"]
        + " ↔ "
        + top["membrane_atom"]
    )

    values = (
        top["occupancy_percent"]
        .to_numpy()
    )

    y = np.arange(
        len(top)
    )

    cm_to_in = 1.0 / 2.54

    plt.figure(
        figsize=(
            14.0 * cm_to_in,
            10.0 * cm_to_in
        ),
        dpi=300
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
        "Top LIG–PMf hydrophobic contacts"
    )

    plt.xlim(
        0,
        max(
            100.0,
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


# ============================================================
# PLOT: TOP MEMBRANE RESIDUES
# ============================================================

def save_residue_plot(df):

    if df.empty:
        return

    top = df.head(
        TOP_N
    ).copy()

    y = np.arange(
        len(top)
    )

    cm_to_in = 1.0 / 2.54

    plt.figure(
        figsize=(
            13.0 * cm_to_in,
            10.0 * cm_to_in
        ),
        dpi=300
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
        "PMf residues forming hydrophobic contacts with LIG"
    )

    plt.xlim(
        0,
        max(
            100.0,
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
# PLOT: LIPID TYPE OCCUPANCY
# ============================================================

def save_lipid_plot(df):

    if df.empty:
        return

    plot_df = df.copy()

    y = np.arange(
        len(plot_df)
    )

    cm_to_in = 1.0 / 2.54

    plt.figure(
        figsize=(
            11.0 * cm_to_in,
            10.0 * cm_to_in
        ),
        dpi=300
    )

    plt.barh(
        y,
        plot_df["occupancy_percent"]
    )

    plt.yticks(
        y,
        plot_df["membrane_resname"]
    )

    plt.xlabel(
        "Occupancy (%)"
    )

    plt.title(
        "Hydrophobic-contact occupancy by PMf component"
    )

    plt.xlim(
        0,
        max(
            100.0,
            plot_df[
                "occupancy_percent"
            ].max() * 1.10
        )
    )

    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.savefig(
        LIPID_OCC_PNG,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 80
    )

    print(
        "LIG <-> PMf HYDROPHOBIC CONTACT ANALYSIS"
    )

    print(
        "SMD 60 A / 1.5 ns"
    )

    print(
        "=" * 80
    )


    # ========================================================
    # CHECK INPUTS
    # ========================================================

    if not os.path.isfile(
        TOPOLOGY
    ):

        raise FileNotFoundError(
            f"Topology not found:\n{TOPOLOGY}"
        )


    if not os.path.isfile(
        TRAJECTORY
    ):

        raise FileNotFoundError(
            f"Trajectory not found:\n{TRAJECTORY}"
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
        "\n[INFO] Output folder:"
    )

    print(
        OUTDIR
    )


    # ========================================================
    # LOAD UNIVERSE
    # ========================================================

    print(
        "\n[INFO] Loading trajectory..."
    )

    u = mda.Universe(
        TOPOLOGY,
        TRAJECTORY
    )


    print(
        f"[INFO] Total atoms: "
        f"{u.atoms.n_atoms:,}"
    )

    print(
        f"[INFO] Total frames: "
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
    # GET LIGAND AND MEMBRANE
    # ========================================================

    ligand_all = u.select_atoms(
        LIG_SEL
    )

    membrane_all = u.select_atoms(
        MEM_SEL
    )


    if ligand_all.n_atoms == 0:

        raise RuntimeError(
            "No LIG atoms were found."
        )


    if membrane_all.n_atoms == 0:

        raise RuntimeError(
            "No PMf membrane atoms were found."
        )


    print(
        f"\n[INFO] Ligand atoms: "
        f"{ligand_all.n_atoms:,}"
    )

    print(
        f"[INFO] PMf membrane atoms: "
        f"{membrane_all.n_atoms:,}"
    )


    # ========================================================
    # IDENTIFY HYDROPHOBIC ATOMS
    # ========================================================

    ligand_hydrophobic_indices = [

        atom.index

        for atom in ligand_all

        if is_hydrophobic_atom(
            atom
        )
    ]


    membrane_hydrophobic_indices = [

        atom.index

        for atom in membrane_all

        if is_hydrophobic_atom(
            atom
        )
    ]


    ligand_hydrophobic = u.atoms[
        ligand_hydrophobic_indices
    ]

    membrane_hydrophobic = u.atoms[
        membrane_hydrophobic_indices
    ]


    print(
        f"\n[INFO] Ligand hydrophobic atoms: "
        f"{ligand_hydrophobic.n_atoms}"
    )

    print(
        f"[INFO] Membrane hydrophobic atoms: "
        f"{membrane_hydrophobic.n_atoms:,}"
    )


    if ligand_hydrophobic.n_atoms == 0:

        raise RuntimeError(
            "No ligand hydrophobic C/S atoms found."
        )


    if membrane_hydrophobic.n_atoms == 0:

        raise RuntimeError(
            "No membrane hydrophobic C/S atoms found."
        )


    print(
        f"\n[INFO] Contact cutoff: "
        f"{CONTACT_CUTOFF_A:.2f} Å"
    )

    print(
        f"[INFO] STEP: {STEP}"
    )


    # ========================================================
    # EVENT STORAGE
    # ========================================================

    event_rows = []

    frame_rows = []


    # ========================================================
    # LOOP THROUGH TRAJECTORY
    # ========================================================

    print(
        "\n[INFO] Calculating hydrophobic contacts..."
    )


    for ts in u.trajectory[::STEP]:

        frame = int(
            ts.frame
        )


        try:

            time_ns = (
                float(ts.time)
                /
                1000.0
            )

        except Exception:

            time_ns = (
                frame
                *
                float(u.trajectory.dt)
                /
                1000.0
            )


        # ----------------------------------------------------
        # Find all hydrophobic atom pairs <= cutoff
        #
        # capped_distance returns local indices for each
        # AtomGroup plus distances
        # ----------------------------------------------------

        pairs, distances = capped_distance(

            ligand_hydrophobic.positions,

            membrane_hydrophobic.positions,

            max_cutoff=
                CONTACT_CUTOFF_A,

            box=ts.dimensions,

            return_distances=True
        )


        # ----------------------------------------------------
        # NUMBER OF CONTACTS
        # ----------------------------------------------------

        if pairs is None:

            n_contacts = 0

        else:

            n_contacts = len(
                pairs
            )


        frame_rows.append({

            "frame":
                frame,

            "time_ns":
                time_ns,

            "hydrophobic_contacts":
                n_contacts,

            "has_hydrophobic_contact":
                int(
                    n_contacts > 0
                )

        })


        # ----------------------------------------------------
        # STORE INDIVIDUAL CONTACT EVENTS
        # ----------------------------------------------------

        if (
            pairs is not None
            and len(pairs) > 0
        ):

            for pair, distance in zip(
                pairs,
                distances
            ):

                lig_local = int(
                    pair[0]
                )

                mem_local = int(
                    pair[1]
                )


                lig_atom = (
                    ligand_hydrophobic[
                        lig_local
                    ]
                )

                mem_atom = (
                    membrane_hydrophobic[
                        mem_local
                    ]
                )


                event_rows.append({

                    "frame":
                        frame,

                    "time_ns":
                        time_ns,

                    "ligand_idx":
                        lig_atom.index,

                    "membrane_idx":
                        mem_atom.index,

                    "ligand_atom":
                        atom_label(
                            lig_atom
                        ),

                    "membrane_atom":
                        atom_label(
                            mem_atom
                        ),

                    "membrane_residue":
                        residue_label(
                            mem_atom
                        ),

                    "membrane_resname":
                        mem_atom.resname,

                    "distance_A":
                        float(
                            distance
                        )

                })


    # ========================================================
    # CREATE DATAFRAMES
    # ========================================================

    per_frame = pd.DataFrame(
        frame_rows
    )

    events = pd.DataFrame(
        event_rows
    )


    sampled_frames = len(
        per_frame
    )


    print(
        f"\n[INFO] Frames analyzed: "
        f"{sampled_frames:,}"
    )


    # ========================================================
    # SAVE PER-FRAME DATA
    # ========================================================

    per_frame.to_csv(
        PER_FRAME_OUT,
        index=False
    )

    save_per_frame_plot(
        per_frame
    )


    # ========================================================
    # GLOBAL STATISTICS
    # ========================================================

    mean_contacts = (
        per_frame[
            "hydrophobic_contacts"
        ].mean()
    )

    sd_contacts = (
        per_frame[
            "hydrophobic_contacts"
        ].std()
    )

    median_contacts = (
        per_frame[
            "hydrophobic_contacts"
        ].median()
    )

    max_contacts = (
        per_frame[
            "hydrophobic_contacts"
        ].max()
    )


    overall_occupancy = (

        100.0
        *
        per_frame[
            "has_hydrophobic_contact"
        ].mean()

    )


    # ========================================================
    # HANDLE ZERO CONTACTS
    # ========================================================

    if events.empty:

        print(
            "\n[INFO] No hydrophobic LIG-PMf "
            "contacts were detected."
        )


        events.to_csv(
            EVENTS_OUT,
            index=False
        )


        with open(
            SUMMARY_OUT,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                "No hydrophobic LIG-PMf contacts detected.\n"
            )


        return


    # ========================================================
    # SAVE ALL CONTACT EVENTS
    # ========================================================

    events.to_csv(
        EVENTS_OUT,
        index=False
    )


    # ========================================================
    # ATOM-PAIR OCCUPANCY
    #
    # One ligand atom - membrane atom pair counted
    # at most once per frame.
    # ========================================================

    pair_events = (

        events

        .drop_duplicates(
            subset=[
                "frame",
                "ligand_idx",
                "membrane_idx"
            ]
        )
    )


    pair_occ = (

        pair_events

        .groupby(
            [
                "ligand_idx",
                "membrane_idx"
            ]
        )

        .agg(

            ligand_atom=(
                "ligand_atom",
                "first"
            ),

            membrane_atom=(
                "membrane_atom",
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

            frames_with_contact=(
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

            distance_min_A=(
                "distance_A",
                "min"
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
            "frames_with_contact"
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


    save_pair_plot(
        pair_occ
    )


    # ========================================================
    # INDIVIDUAL MEMBRANE RESIDUE OCCUPANCY
    #
    # Percentage of frames in which the ligand has at least
    # one hydrophobic contact with that residue.
    # ========================================================

    residue_events = (

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

        residue_events

        .groupby(
            [
                "membrane_residue",
                "membrane_resname"
            ]
        )

        .agg(

            frames_with_contact=(
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
            "frames_with_contact"
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


    save_residue_plot(
        residue_occ
    )


    # ========================================================
    # OCCUPANCY BY MEMBRANE COMPONENT / LIPID TYPE
    # ========================================================

    lipid_events = (

        events[
            [
                "frame",
                "membrane_resname"
            ]
        ]

        .drop_duplicates()
    )


    lipid_occ = (

        lipid_events

        .groupby(
            "membrane_resname"
        )

        .agg(

            frames_with_contact=(
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
            "frames_with_contact"
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


    save_lipid_plot(
        lipid_occ
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    total_contact_events = len(
        events
    )

    unique_pairs = len(
        pair_occ
    )


    with open(
        SUMMARY_OUT,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "LIG <-> PMf HYDROPHOBIC CONTACT ANALYSIS\n"
        )

        f.write(
            "SMD 60 A / 1.5 ns\n"
        )

        f.write(
            "=" * 60
            + "\n\n"
        )


        f.write(
            f"Topology:\n{TOPOLOGY}\n\n"
        )

        f.write(
            f"Trajectory:\n{TRAJECTORY}\n\n"
        )


        f.write(
            "Operational definition:\n"
        )

        f.write(
            "C/S nonpolar-heavy-atom contact\n"
        )

        f.write(
            f"Distance cutoff: "
            f"{CONTACT_CUTOFF_A:.2f} Å\n\n"
        )


        f.write(
            f"Trajectory frames: "
            f"{u.trajectory.n_frames}\n"
        )

        f.write(
            f"Frames sampled: "
            f"{sampled_frames}\n\n"
        )


        f.write(
            f"Ligand hydrophobic atoms: "
            f"{ligand_hydrophobic.n_atoms}\n"
        )

        f.write(
            f"Membrane hydrophobic atoms: "
            f"{membrane_hydrophobic.n_atoms}\n\n"
        )


        f.write(
            "CONTACTS PER FRAME\n"
        )

        f.write(
            "-" * 40
            + "\n"
        )

        f.write(
            f"Mean ± SD: "
            f"{mean_contacts:.4f} "
            f"± {sd_contacts:.4f}\n"
        )

        f.write(
            f"Median: "
            f"{median_contacts:.4f}\n"
        )

        f.write(
            f"Maximum: "
            f"{max_contacts}\n\n"
        )


        f.write(
            "OVERALL CONTACT OCCUPANCY\n"
        )

        f.write(
            "-" * 40
            + "\n"
        )

        f.write(
            "Frames with >=1 hydrophobic "
            "LIG-PMf contact:\n"
        )

        f.write(
            f"{overall_occupancy:.3f}%\n\n"
        )


        f.write(
            f"Total contact events: "
            f"{total_contact_events}\n"
        )

        f.write(
            f"Unique atom pairs: "
            f"{unique_pairs}\n\n"
        )


        f.write(
            "TOP HYDROPHOBIC CONTACT PAIRS\n"
        )

        f.write(
            "-" * 40
            + "\n"
        )


        for _, row in (
            pair_occ
            .head(TOP_N)
            .iterrows()
        ):

            f.write(
                f"{row['ligand_atom']} <-> "
                f"{row['membrane_atom']}\n"
            )

            f.write(
                f"  Membrane residue: "
                f"{row['membrane_residue']}\n"
            )

            f.write(
                f"  Occupancy: "
                f"{row['occupancy_percent']:.3f}%\n"
            )

            f.write(
                f"  Mean distance: "
                f"{row['distance_mean_A']:.3f} Å\n"
            )

            f.write(
                f"  Minimum distance: "
                f"{row['distance_min_A']:.3f} Å\n\n"
            )


    # ========================================================
    # CONSOLE OUTPUT
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "RESULTS"
    )

    print(
        "=" * 80
    )


    print(
        f"\nMean hydrophobic contacts/frame: "
        f"{mean_contacts:.3f} "
        f"± {sd_contacts:.3f}"
    )

    print(
        f"Median contacts/frame: "
        f"{median_contacts:.3f}"
    )

    print(
        f"Maximum contacts/frame: "
        f"{max_contacts}"
    )


    print(
        f"\nFrames with >=1 hydrophobic contact: "
        f"{overall_occupancy:.3f}%"
    )


    print(
        f"\nUnique hydrophobic atom pairs: "
        f"{unique_pairs}"
    )


    print(
        "\nTop hydrophobic atom pairs:"
    )


    print(

        pair_occ[
            [
                "ligand_atom",
                "membrane_atom",
                "membrane_resname",
                "occupancy_percent",
                "distance_mean_A",
                "distance_min_A"
            ]
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
        "\nOccupancy by PMf component:"
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
        + "=" * 80
    )

    print(
        "ANALYSIS COMPLETE"
    )

    print(
        "=" * 80
    )


    print(
        f"\nAll outputs saved in:\n"
        f"{OUTDIR}"
    )


    print(
        """
Files:

hydrophobic_contacts_LIG_PMF_SMD_all_events.csv

hydrophobic_contacts_LIG_PMF_SMD_per_frame.csv

hydrophobic_contacts_LIG_PMF_SMD_pair_occupancy.csv

hydrophobic_contacts_LIG_PMF_SMD_residue_occupancy.csv

hydrophobic_contacts_LIG_PMF_SMD_lipid_type_occupancy.csv

hydrophobic_contacts_LIG_PMF_SMD_summary.txt

hydrophobic_contacts_LIG_PMF_SMD_per_frame.png

hydrophobic_contacts_LIG_PMF_SMD_top10_pair_occupancy.png

hydrophobic_contacts_LIG_PMF_SMD_top10_residue_occupancy.png

hydrophobic_contacts_LIG_PMF_SMD_lipid_type_occupancy.png
"""
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()