import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

# Parent folder containing the NAMD energy log
base_dir = (
    r"D:\fcm-uerj\artigos-e-producao"
    r"\2027_candida-MMV687807-675968_daniel"
    r"\PMf_MMV675968_SMD"
)

# Input energy log
log_file = os.path.join(
    base_dir,
    "energias-SMD-PMf-MMV675968-60A-1p5ns.log"
)

# Folder where ALL outputs will be saved
output_dir = os.path.join(
    base_dir,
    "NAMD_PMf_MMV675968_staged"
)

os.makedirs(output_dir, exist_ok=True)


# ============================================================
# OUTPUT FILES
# ============================================================

output_csv = os.path.join(
    output_dir,
    "MMV675968_energy_data.csv"
)

output_summary = os.path.join(
    output_dir,
    "MMV675968_energy_summary.csv"
)

output_combined = os.path.join(
    output_dir,
    "MMV675968_energy_components.png"
)

output_global = os.path.join(
    output_dir,
    "MMV675968_total_potential_kinetic.png"
)

output_temperature = os.path.join(
    output_dir,
    "MMV675968_temperature.png"
)

output_pressure = os.path.join(
    output_dir,
    "MMV675968_pressure.png"
)

output_volume = os.path.join(
    output_dir,
    "MMV675968_volume.png"
)


# ============================================================
# NAMD SETTINGS
# ============================================================

# NAMD integration timestep in femtoseconds
DT_FS = 2.0


# ============================================================
# FUNCTION TO PARSE NAMD NUMBERS
# ============================================================

def parse_namd_number(value):
    """
    Convert NAMD output values to float.

    Handles:
    - decimal values
    - scientific notation
    - NaN
    - overflow values printed as *****
    """

    value = value.strip()

    if "*" in value:
        return np.nan

    try:
        return float(value)

    except ValueError:
        return np.nan


# ============================================================
# CHECK FILES
# ============================================================

print("=" * 75)
print("NAMD ENERGY ANALYSIS - MMV675968")
print("=" * 75)

print("\nInput log:")
print(log_file)

print("\nOutput folder:")
print(output_dir)


if not os.path.isfile(log_file):
    raise FileNotFoundError(
        f"\nERROR: Energy log not found:\n{log_file}"
    )


# ============================================================
# PARSE ETITLE AND ENERGY
# ============================================================

energy_columns = None
records = []

with open(
    log_file,
    "r",
    encoding="utf-8",
    errors="ignore"
) as f:

    for line in f:

        stripped = line.strip()

        # ----------------------------------------------------
        # NAMD ENERGY header
        # ----------------------------------------------------

        if stripped.startswith("ETITLE:"):

            parts = stripped.split()

            detected_columns = parts[1:]

            if detected_columns:
                energy_columns = detected_columns

        # ----------------------------------------------------
        # ENERGY values
        # ----------------------------------------------------

        elif stripped.startswith("ENERGY:"):

            if energy_columns is None:
                continue

            parts = stripped.split()[1:]

            values = [
                parse_namd_number(value)
                for value in parts
            ]

            # If fewer values than columns, fill with NaN
            if len(values) < len(energy_columns):

                values += (
                    [np.nan]
                    * (len(energy_columns) - len(values))
                )

            # If unexpected extra values exist, truncate
            elif len(values) > len(energy_columns):

                values = values[:len(energy_columns)]

            record = dict(
                zip(
                    energy_columns,
                    values
                )
            )

            records.append(record)


# ============================================================
# CHECK EXTRACTION
# ============================================================

if not records:

    raise RuntimeError(
        "\nERROR: No ENERGY records could be extracted."
    )


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(records)


print("\nDetected NAMD columns:")

for i, col in enumerate(df.columns, start=1):
    print(f"{i:02d}. {col}")


print(
    f"\nENERGY records extracted: {len(df):,}"
)


# ============================================================
# ADD SIMULATION TIME
# ============================================================

if "TS" in df.columns:

    time_fs = df["TS"] * DT_FS

    df.insert(
        1,
        "TIME_FS",
        time_fs
    )

    df.insert(
        2,
        "TIME_PS",
        time_fs / 1000.0
    )

    df.insert(
        3,
        "TIME_NS",
        time_fs / 1_000_000.0
    )

    print(
        f"\nAssumed NAMD timestep: {DT_FS} fs"
    )

    print(
        "Maximum simulation time represented in log: "
        f"{df['TIME_NS'].max():.6f} ns"
    )


# ============================================================
# SAVE COMPLETE ENERGY TABLE
# ============================================================

df.to_csv(
    output_csv,
    index=False,
    sep=";",
    decimal=",",
    encoding="utf-8-sig"
)


print(
    f"\n[OK] Complete energy data saved:\n"
    f"{output_csv}"
)


# ============================================================
# SUMMARY STATISTICS
# ============================================================

numeric_cols = df.select_dtypes(
    include=[np.number]
).columns


summary = pd.DataFrame({
    "N": df[numeric_cols].count(),
    "Mean": df[numeric_cols].mean(),
    "SD": df[numeric_cols].std(),
    "Median": df[numeric_cols].median(),
    "Minimum": df[numeric_cols].min(),
    "Maximum": df[numeric_cols].max()
})


summary.to_csv(
    output_summary,
    sep=";",
    decimal=",",
    encoding="utf-8-sig"
)


print(
    f"\n[OK] Summary statistics saved:\n"
    f"{output_summary}"
)


# ============================================================
# X AXIS
# ============================================================

if "TIME_NS" in df.columns:

    x = df["TIME_NS"]
    x_label = "Time (ns)"

elif "TS" in df.columns:

    x = df["TS"]
    x_label = "Timestep"

else:

    x = np.arange(len(df))
    x_label = "Frame"


# ============================================================
# PARAMETERS TO PLOT
# ============================================================

exclude_columns = {
    "TS",
    "TIME_FS",
    "TIME_PS",
    "TIME_NS"
}


plot_columns = [
    col
    for col in numeric_cols
    if col not in exclude_columns
]


# ============================================================
# INDIVIDUAL PLOTS
# ============================================================

print("\nCreating individual parameter plots...")


energy_terms = {
    "BOND",
    "ANGLE",
    "DIHED",
    "IMPRP",
    "ELECT",
    "VDW",
    "BOUNDARY",
    "MISC",
    "KINETIC",
    "TOTAL",
    "POTENTIAL",
    "TOTAL2",
    "TOTAL3"
}

temperature_terms = {
    "TEMP",
    "TEMPAVG"
}

pressure_terms = {
    "PRESSURE",
    "GPRESS",
    "PRESSAVG",
    "GPRESSAVG"
}


for col in plot_columns:

    values = df[col]

    if values.notna().sum() == 0:
        continue

    plt.figure(
        figsize=(5, 4),
        dpi=300
    )

    plt.plot(
        x,
        values,
        linewidth=0.8
    )

    plt.xlabel(x_label)


    if col in energy_terms:

        ylabel = f"{col} (kcal/mol)"

    elif col in temperature_terms:

        ylabel = f"{col} (K)"

    elif col in pressure_terms:

        ylabel = f"{col} (bar)"

    elif col == "VOLUME":

        ylabel = "Volume (Å³)"

    else:

        ylabel = col


    plt.ylabel(ylabel)
    plt.title(col)

    plt.tight_layout()


    safe_name = re.sub(
        r"[^A-Za-z0-9_-]",
        "_",
        col
    )


    output_plot = os.path.join(
        output_dir,
        f"MMV675968_{safe_name}.png"
    )


    plt.savefig(
        output_plot,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


print(
    "\n[OK] Individual plots completed."
)


# ============================================================
# COMBINED ENERGY COMPONENTS
# ============================================================

main_energy_terms = [
    "BOND",
    "ANGLE",
    "DIHED",
    "IMPRP",
    "ELECT",
    "VDW",
    "BOUNDARY",
    "MISC"
]


available_main_terms = [
    col
    for col in main_energy_terms
    if col in df.columns
]


if available_main_terms:

    plt.figure(
        figsize=(7, 5),
        dpi=300
    )

    for col in available_main_terms:

        plt.plot(
            x,
            df[col],
            label=col,
            linewidth=0.8
        )


    plt.xlabel(x_label)
    plt.ylabel("Energy (kcal/mol)")

    plt.title(
        "NAMD Energy Components - MMV675968"
    )

    plt.legend(
        fontsize=8,
        frameon=False
    )

    plt.tight_layout()

    plt.savefig(
        output_combined,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    print(
        f"\n[OK] Combined energy plot:\n"
        f"{output_combined}"
    )


# ============================================================
# TOTAL / POTENTIAL / KINETIC
# ============================================================

global_energy_terms = [
    "TOTAL",
    "POTENTIAL",
    "KINETIC",
    "TOTAL2",
    "TOTAL3"
]


available_global_terms = [
    col
    for col in global_energy_terms
    if col in df.columns
]


if available_global_terms:

    plt.figure(
        figsize=(7, 5),
        dpi=300
    )

    for col in available_global_terms:

        plt.plot(
            x,
            df[col],
            label=col,
            linewidth=0.8
        )


    plt.xlabel(x_label)
    plt.ylabel("Energy (kcal/mol)")

    plt.title(
        "Total, Potential and Kinetic Energy"
    )

    plt.legend(
        fontsize=8,
        frameon=False
    )

    plt.tight_layout()

    plt.savefig(
        output_global,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# TEMPERATURE
# ============================================================

available_temperature = [
    col
    for col in [
        "TEMP",
        "TEMPAVG"
    ]
    if col in df.columns
]


if available_temperature:

    plt.figure(
        figsize=(6, 4),
        dpi=300
    )

    for col in available_temperature:

        plt.plot(
            x,
            df[col],
            label=col,
            linewidth=0.8
        )


    plt.xlabel(x_label)
    plt.ylabel("Temperature (K)")
    plt.title("Temperature")

    plt.legend(
        frameon=False
    )

    plt.tight_layout()

    plt.savefig(
        output_temperature,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# PRESSURE
# ============================================================

available_pressure = [
    col
    for col in [
        "PRESSURE",
        "GPRESS",
        "PRESSAVG",
        "GPRESSAVG"
    ]
    if col in df.columns
]


if available_pressure:

    plt.figure(
        figsize=(6, 4),
        dpi=300
    )

    for col in available_pressure:

        plt.plot(
            x,
            df[col],
            label=col,
            linewidth=0.8
        )


    plt.xlabel(x_label)
    plt.ylabel("Pressure (bar)")
    plt.title("Pressure")

    plt.legend(
        frameon=False
    )

    plt.tight_layout()

    plt.savefig(
        output_pressure,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# VOLUME
# ============================================================

if "VOLUME" in df.columns:

    plt.figure(
        figsize=(6, 4),
        dpi=300
    )

    plt.plot(
        x,
        df["VOLUME"],
        linewidth=0.8
    )

    plt.xlabel(x_label)
    plt.ylabel("Volume (Å³)")
    plt.title("Simulation Cell Volume")

    plt.tight_layout()

    plt.savefig(
        output_volume,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 75)
print("ANALYSIS COMPLETE")
print("=" * 75)

print(
    f"""
INPUT ENERGY LOG:

{log_file}


ALL ANALYSIS OUTPUTS WERE SAVED TO:

{output_dir}


Main files:

MMV675968_energy_data.csv
MMV675968_energy_summary.csv

MMV675968_energy_components.png
MMV675968_total_potential_kinetic.png
MMV675968_temperature.png
MMV675968_pressure.png
MMV675968_volume.png

Individual parameter figures:
MMV675968_BOND.png
MMV675968_ANGLE.png
MMV675968_DIHED.png
MMV675968_IMPRP.png
MMV675968_ELECT.png
MMV675968_VDW.png
etc.

ENERGY records extracted:
{len(df):,}

NAMD parameters detected:
{len(energy_columns)}
"""
)