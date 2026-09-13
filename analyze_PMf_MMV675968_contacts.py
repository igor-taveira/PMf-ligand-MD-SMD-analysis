import os

import MDAnalysis as mda
import matplotlib.pyplot as plt
import numpy as np
from MDAnalysis.lib.distances import distance_array

# === CONFIGURATION ===
work_dir = (
    r"D:\fcm-uerj\artigos-e-producao"
    r"\2027_candida-MMV687807-675968_daniel"
    r"\PMf_MMV675968_SMD"
)

trajectory_dir = os.path.join(
    work_dir,
    "NAMD_PMf_MMV675968_staged"
)

# The PSF is in the main folder, according to the directory listing
psf = os.path.join(
    work_dir,
    "PMf_LIG_nooverlap.psf"
)

# The trajectory is inside NAMD_PMf_MMV675968_staged
dcd = os.path.join(
    trajectory_dir,
    "PMf_MMV675968_staged.dcd"
)

output_txt = os.path.join(
    trajectory_dir,
    "PMf_MMV675968_membrane_contacts.txt"
)

output_png = os.path.join(
    trajectory_dir,
    "PMf_MMV675968_membrane_contacts.png"
)

cutoff = 4.5  # Å

# === VERIFY INPUT FILES ===
if not os.path.isfile(psf):
    raise FileNotFoundError(f"PSF file not found:\n{psf}")

if not os.path.isfile(dcd):
    raise FileNotFoundError(f"DCD file not found:\n{dcd}")

# === LOAD SYSTEM ===
print("Loading topology and trajectory...")
u = mda.Universe(psf, dcd)

# === DEFINE SELECTIONS ===
# Conventional lipids are in MEMB; fungal glycosphingolipids are
# distributed across GLPA1–GLPA54.
membrane = u.select_atoms(
    "(segid MEMB) or (resname BMAN CER160)"
)

# MMV675968
ligand = u.select_atoms(
    "segid L and resname LIG"
)

# === VALIDATE SELECTIONS ===
if membrane.n_atoms == 0:
    available_segids = sorted(set(u.atoms.segids))

    raise ValueError(
        "No membrane atoms were found using segids L11–L24.\n"
        f"Available segids: {available_segids}"
    )

if ligand.n_atoms == 0:
    available_resnames = sorted(set(u.atoms.resnames))
    available_segids = sorted(set(u.atoms.segids))

    raise ValueError(
        "No ligand atoms were found using segid/resname LIG.\n"
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

print("Calculating MMV675968–membrane contacts...")

for index, ts in enumerate(u.trajectory):
    distances = distance_array(
        ligand.positions,
        membrane.positions,
        box=ts.dimensions
    )

    # Number of ligand–membrane atom pairs closer than the cutoff
    n_contacts = int(np.count_nonzero(distances < cutoff))

    frames.append(ts.frame)
    times_ns.append(ts.time / 1000.0)  # MDAnalysis time: ps → ns
    contact_counts.append(n_contacts)

    if index % 100 == 0 or index == len(u.trajectory) - 1:
        print(
            f"Processed frame {index + 1}/{len(u.trajectory)} "
            f"({ts.time / 1000.0:.3f} ns)"
        )

# === SAVE RESULTS ===
results = np.column_stack(
    (frames, times_ns, contact_counts)
)

np.savetxt(
    output_txt,
    results,
    fmt=["%d", "%.6f", "%d"],
    header="Frame Time_ns Contacts",
    comments=""
)

# === PLOT ===
plt.figure(figsize=(4, 4))

plt.plot(
    times_ns,
    contact_counts,
    color="darkblue",
    linewidth=1.2
)

plt.xlabel("Time (ns)")
plt.ylabel("Number of contacts")
plt.title(f"PMf–MMV675968 contacts\ncutoff = {cutoff:.1f} Å")

plt.tight_layout()
plt.savefig(output_png, dpi=300, bbox_inches="tight")
plt.close()

print("\nAnalysis completed.")
print(f"Numerical results:\n{output_txt}")
print(f"Figure:\n{output_png}")