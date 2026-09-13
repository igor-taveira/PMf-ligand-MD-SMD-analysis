import os

import MDAnalysis as mda
import matplotlib.pyplot as plt
import numpy as np
from MDAnalysis.lib.distances import capped_distance


# === CONFIGURATION ===
work_dir = (
    r"D:\fcm-uerj\artigos-e-producao"
    r"\2027_candida-MMV687807-675968_daniel"
    r"\PMf_MMV675968_SMD"
)

trajectory_dir = os.path.join(
    work_dir,
    "NAMD_SMD_MMV675968_60A-1p5ns",
)

# Topology is stored in the main PMf_MMV675968_SMD directory.
psf = os.path.join(work_dir, "PMf_LIG_nooverlap.psf")

# Forward SMD trajectory: 60 Angstrom over 1.5 ns.
dcd = os.path.join(
    trajectory_dir,
    "PMf_MMV675968_SMD_60A_1p5ns.dcd",
)

output_txt = os.path.join(
    trajectory_dir,
    "PMf_MMV675968_SMD_60A_1p5ns_contacts.txt",
)

output_png = os.path.join(
    trajectory_dir,
    "PMf_MMV675968_SMD_60A_1p5ns_contacts.png",
)

cutoff = 4.5  # Angstrom


# === VERIFY INPUT FILES ===
if not os.path.isfile(psf):
    raise FileNotFoundError(f"PSF file not found:\n{psf}")

if not os.path.isfile(dcd):
    raise FileNotFoundError(f"DCD file not found:\n{dcd}")


# === LOAD SYSTEM ===
print("Loading topology and SMD trajectory...")
u = mda.Universe(psf, dcd)


# === DEFINE SELECTIONS ===
# Conventional fungal-membrane lipids are in segid MEMB.
# Glycosphingolipids are represented by BMAN and CER160 residues
# distributed across segids GLPA1-GLPA54.
membrane = u.select_atoms(
    "(segid MEMB) or (resname BMAN CER160)"
)

# MMV675968 is residue LIG in segment L.
ligand = u.select_atoms(
    "segid L and resname LIG"
)


# === VALIDATE SELECTIONS ===
if membrane.n_atoms == 0:
    available_segids = sorted(set(u.atoms.segids))
    available_resnames = sorted(set(u.atoms.resnames))

    raise ValueError(
        "No membrane atoms were found.\n"
        f"Available segids: {available_segids}\n"
        f"Available resnames: {available_resnames}"
    )

if ligand.n_atoms == 0:
    available_segids = sorted(set(u.atoms.segids))
    available_resnames = sorted(set(u.atoms.resnames))

    raise ValueError(
        "No MMV675968 atoms were found using 'segid L and resname LIG'.\n"
        f"Available segids: {available_segids}\n"
        f"Available resnames: {available_resnames}"
    )

print(f"Total system atoms: {u.atoms.n_atoms}")
print(f"Membrane atoms: {membrane.n_atoms}")
print(f"MMV675968 atoms: {ligand.n_atoms}")
print(f"Trajectory frames: {len(u.trajectory)}")


# === CALCULATE CONTACTS FRAME BY FRAME ===
frames = []
times_ns = []
contact_counts = []

print(f"Calculating contacts using a {cutoff:.1f}-Angstrom cutoff...")

for index, ts in enumerate(u.trajectory):
    # capped_distance returns only atom pairs within the cutoff and avoids
    # constructing a complete ligand-by-membrane distance matrix.
    contacting_pairs = capped_distance(
        ligand.positions,
        membrane.positions,
        max_cutoff=cutoff,
        box=ts.dimensions,
        return_distances=False,
    )

    n_contacts = int(contacting_pairs.shape[0])

    frames.append(ts.frame)
    times_ns.append(ts.time / 1000.0)  # ps to ns
    contact_counts.append(n_contacts)

    if index % 50 == 0 or index == len(u.trajectory) - 1:
        print(
            f"Processed frame {index + 1}/{len(u.trajectory)} "
            f"({ts.time / 1000.0:.4f} ns)"
        )


# === SAVE NUMERICAL RESULTS ===
results = np.column_stack(
    (frames, times_ns, contact_counts)
)

np.savetxt(
    output_txt,
    results,
    fmt=["%d", "%.6f", "%d"],
    delimiter="\t",
    header="Frame\tTime_ns\tContacts",
    comments="",
)


# === PLOT CONTACTS OVER TIME ===
plt.figure(figsize=(4, 4))

plt.plot(
    times_ns,
    contact_counts,
    color="darkblue",
    linewidth=1.2,
)

plt.xlabel("Time (ns)")
plt.ylabel("Number of atom-pair contacts")
plt.title(
    "PMf-MMV675968 contacts during SMD\n"
    f"60 Angstrom/1.5 ns; cutoff = {cutoff:.1f} Angstrom"
)

plt.tight_layout()
plt.savefig(output_png, dpi=300, bbox_inches="tight")
plt.close()


# === FINAL SUMMARY ===
contact_array = np.asarray(contact_counts, dtype=float)

print("\nAnalysis completed successfully.")
print(f"Mean contacts: {contact_array.mean():.2f}")
print(f"Minimum contacts: {int(contact_array.min())}")
print(f"Maximum contacts: {int(contact_array.max())}")
print(f"Numerical results:\n{output_txt}")
print(f"Figure:\n{output_png}")
