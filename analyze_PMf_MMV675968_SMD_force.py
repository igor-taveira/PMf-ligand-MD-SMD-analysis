import os

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================
work_dir = (
    r"D:\fcm-uerj\artigos-e-producao"
    r"\2027_candida-MMV687807-675968_daniel"
    r"\PMf_MMV675968_SMD"
)

trajectory_dir = os.path.join(
    work_dir,
    "NAMD_SMD_MMV675968_60A-1p5ns"
)

# NAMD log containing lines beginning with "SMD"
log_file = os.path.join(
    work_dir,
    "energias-SMD-PMf-MMV675968-60A-1p5ns.log"
)

# Configuration file used for the simulation
conf_file = os.path.join(
    work_dir,
    "SMD_PMF_MMV675968_60A_1p5ns_SMDfolder.conf"
)

output_txt = os.path.join(
    trajectory_dir,
    "PMf_MMV675968_SMD_60A_1p5ns_force_vs_time.txt"
)

output_png = os.path.join(
    trajectory_dir,
    "PMf_MMV675968_SMD_60A_1p5ns_force_vs_time.png"
)

# Used only if the values cannot be extracted from the configuration file
default_timestep_fs = 2.0
default_pull_direction = np.array([0.0, 0.0, 1.0])

# Number of points used for the moving-average curve
smoothing_window = 25


# ============================================================
# READ TIMESTEP AND SMD DIRECTION FROM CONFIGURATION
# ============================================================
def read_smd_configuration(filename):
    timestep_fs = None
    pull_direction = None

    if not os.path.isfile(filename):
        print(
            "Warning: configuration file was not found. "
            "Default SMD parameters will be used."
        )
        return timestep_fs, pull_direction

    with open(filename, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            # Remove comments
            line = line.split("#", 1)[0].strip()

            if not line:
                continue

            fields = line.split()
            keyword = fields[0].lower()

            if keyword == "timestep" and len(fields) >= 2:
                try:
                    timestep_fs = float(fields[1])
                except ValueError:
                    pass

            elif keyword == "smddir" and len(fields) >= 4:
                try:
                    pull_direction = np.array(
                        [
                            float(fields[1]),
                            float(fields[2]),
                            float(fields[3]),
                        ],
                        dtype=float,
                    )
                except ValueError:
                    pass

    return timestep_fs, pull_direction


# ============================================================
# READ SMD DATA FROM NAMD LOG
# ============================================================
def read_smd_log(filename):
    """
    Expected NAMD format:

    SMD step COM_x COM_y COM_z Force_x Force_y Force_z

    Positions are in Angstrom and forces are already in pN.
    """
    records_by_step = {}

    with open(filename, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            fields = line.split()

            if len(fields) < 8 or fields[0] != "SMD":
                continue

            try:
                step = int(float(fields[1]))

                com_x = float(fields[2])
                com_y = float(fields[3])
                com_z = float(fields[4])

                force_x = float(fields[5])
                force_y = float(fields[6])
                force_z = float(fields[7])

            except ValueError:
                continue

            # If a restarted log contains a repeated timestep,
            # retain the last occurrence.
            records_by_step[step] = (
                step,
                com_x,
                com_y,
                com_z,
                force_x,
                force_y,
                force_z,
            )

    if not records_by_step:
        raise ValueError(
            "No valid SMD records were found in the log file.\n"
            "The script expects lines formatted as:\n"
            "SMD step COM_x COM_y COM_z Force_x Force_y Force_z"
        )

    records = [
        records_by_step[step]
        for step in sorted(records_by_step)
    ]

    return np.asarray(records, dtype=float)


# ============================================================
# VERIFY INPUT
# ============================================================
if not os.path.isfile(log_file):
    raise FileNotFoundError(
        f"NAMD log file not found:\n{log_file}"
    )

os.makedirs(trajectory_dir, exist_ok=True)


# ============================================================
# OBTAIN SIMULATION PARAMETERS
# ============================================================
timestep_fs, pull_direction = read_smd_configuration(conf_file)

if timestep_fs is None:
    timestep_fs = default_timestep_fs
    print(
        f"Could not read 'timestep'; using {timestep_fs:.3f} fs."
    )

if pull_direction is None:
    pull_direction = default_pull_direction.copy()
    print(
        "Could not read 'SMDDir'; using direction "
        f"{pull_direction.tolist()}."
    )

direction_norm = np.linalg.norm(pull_direction)

if direction_norm == 0:
    raise ValueError("SMDDir cannot be a zero vector.")

# NAMD normalizes SMDDir internally
pull_direction = pull_direction / direction_norm

print(f"Timestep: {timestep_fs:.4f} fs")
print(
    "Normalized pulling direction: "
    f"({pull_direction[0]:.6f}, "
    f"{pull_direction[1]:.6f}, "
    f"{pull_direction[2]:.6f})"
)


# ============================================================
# EXTRACT FORCE DATA
# ============================================================
print("Reading SMD force records...")
data = read_smd_log(log_file)

steps = data[:, 0].astype(int)
com_positions = data[:, 1:4]
force_vectors = data[:, 4:7]

# Relative time, starting at zero
time_ns = (
    (steps - steps[0]) * timestep_fs / 1_000_000.0
)

# Force projected onto the normalized pulling direction
force_parallel_pn = np.dot(
    force_vectors,
    pull_direction
)

# Total magnitude of the SMD force vector
force_magnitude_pn = np.linalg.norm(
    force_vectors,
    axis=1
)

print(f"SMD records found: {len(steps)}")
print(f"Initial timestep: {steps[0]}")
print(f"Final timestep: {steps[-1]}")
print(f"Analyzed duration: {time_ns[-1]:.6f} ns")


# ============================================================
# MOVING AVERAGE
# ============================================================
window = min(smoothing_window, len(force_parallel_pn))

# Use an odd window so it is centered correctly
if window % 2 == 0:
    window -= 1

if window >= 3:
    kernel = np.ones(window, dtype=float) / window

    smoothed_force = np.convolve(
        force_parallel_pn,
        kernel,
        mode="valid"
    )

    half_window = window // 2

    smoothed_time = time_ns[
        half_window:len(time_ns) - half_window
    ]

else:
    smoothed_force = force_parallel_pn.copy()
    smoothed_time = time_ns.copy()


# ============================================================
# SAVE NUMERICAL RESULTS
# ============================================================
results = np.column_stack(
    (
        steps,
        time_ns,
        com_positions,
        force_vectors,
        force_parallel_pn,
        force_magnitude_pn,
    )
)

header = (
    "Step\tTime_ns\t"
    "COM_X_A\tCOM_Y_A\tCOM_Z_A\t"
    "Force_X_pN\tForce_Y_pN\tForce_Z_pN\t"
    "Force_parallel_pN\tForce_magnitude_pN"
)

np.savetxt(
    output_txt,
    results,
    delimiter="\t",
    fmt=[
        "%d",
        "%.8f",
        "%.6f",
        "%.6f",
        "%.6f",
        "%.6f",
        "%.6f",
        "%.6f",
        "%.6f",
        "%.6f",
    ],
    header=header,
    comments="",
)


# ============================================================
# PLOT FORCE VS TIME
# ============================================================
plt.figure(figsize=(5, 4))

plt.plot(
    time_ns,
    force_parallel_pn,
    color="lightsteelblue",
    linewidth=0.8,
    alpha=0.75,
    label="Instantaneous force",
)

if window >= 3:
    plt.plot(
        smoothed_time,
        smoothed_force,
        color="darkblue",
        linewidth=1.6,
        label=f"Moving average ({window} points)",
    )

plt.axhline(
    0,
    color="black",
    linewidth=0.7,
    linestyle="--",
)

plt.xlabel("Time (ns)")
plt.ylabel("Force along pulling direction (pN)")
plt.title("MMV675968 force during SMD\n60 Å over 1.5 ns")
plt.legend(frameon=False, fontsize=8)

plt.tight_layout()
plt.savefig(
    output_png,
    dpi=300,
    bbox_inches="tight",
)
plt.close()


# ============================================================
# SUMMARY
# ============================================================
maximum_index = np.argmax(force_parallel_pn)
maximum_absolute_index = np.argmax(
    np.abs(force_parallel_pn)
)

print("\nAnalysis completed successfully.")

print(
    "Mean force along pulling direction: "
    f"{np.mean(force_parallel_pn):.2f} pN"
)

print(
    "Maximum force along pulling direction: "
    f"{force_parallel_pn[maximum_index]:.2f} pN "
    f"at {time_ns[maximum_index]:.6f} ns"
)

print(
    "Maximum absolute projected force: "
    f"{abs(force_parallel_pn[maximum_absolute_index]):.2f} pN "
    f"at {time_ns[maximum_absolute_index]:.6f} ns"
)

print(f"\nNumerical results:\n{output_txt}")
print(f"\nFigure:\n{output_png}")