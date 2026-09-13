import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# SMD BOUND vs PMf UNBOUND ENERGY DIFFERENCE
#
# Definition:
#
#   DELTA = SMD BOUND - PMf UNBOUND
#
#   BOUND:
#       PMf + MMV675968 during 60 A / 1.5 ns SMD
#
#   UNBOUND:
#       PMf alone
# ============================================================


# ============================================================
# PROJECT PATH
# ============================================================

project_dir = (
    r"D:\fcm-uerj\artigos-e-producao"
    r"\2027_candida-MMV687807-675968_daniel"
)


# ============================================================
# BOUND SYSTEM
# SMD PMf + MMV675968
# ============================================================

bound_parent_dir = os.path.join(
    project_dir,
    "PMf_MMV675968_SMD"
)

bound_log = os.path.join(
    bound_parent_dir,
    "energias-SMD-PMf-MMV675968-60A-1p5ns.log"
)


# ============================================================
# UNBOUND SYSTEM
# PMf alone
# ============================================================

unbound_csv = os.path.join(
    project_dir,
    "PMf",
    "NAMD_PMf_staged",
    "PMf_energy_data.csv"
)


# ============================================================
# WORKING / OUTPUT FOLDER
# ============================================================

working_dir = os.path.join(
    bound_parent_dir,
    "NAMD_SMD_MMV675968_60A-1p5ns"
)

output_dir = os.path.join(
    working_dir,
    "energy_difference_SMD_bound_minus_unbound"
)

os.makedirs(
    output_dir,
    exist_ok=True
)


# ============================================================
# OUTPUT FILES
# ============================================================

output_bound_csv = os.path.join(
    output_dir,
    "MMV675968_SMD_bound_energy_data.csv"
)

output_complete_csv = os.path.join(
    output_dir,
    "MMV675968_SMD_bound_minus_unbound_energy_difference.csv"
)

output_summary_csv = os.path.join(
    output_dir,
    "MMV675968_SMD_bound_minus_unbound_summary.csv"
)

output_components_png = os.path.join(
    output_dir,
    "MMV675968_SMD_delta_energy_components.png"
)

output_global_png = os.path.join(
    output_dir,
    "MMV675968_SMD_delta_global_energies.png"
)

output_nonbonded_png = os.path.join(
    output_dir,
    "MMV675968_SMD_delta_electrostatic_vdw.png"
)

output_mean_png = os.path.join(
    output_dir,
    "MMV675968_SMD_mean_energy_differences.png"
)


# ============================================================
# SETTINGS
# ============================================================

# NAMD timestep in femtoseconds
DT_FS = 2.0

# Rolling average for plots
ROLLING_WINDOW = 50

# Plot raw data together with rolling average
PLOT_RAW = True


# ============================================================
# HELPER: PARSE NAMD NUMBER
# ============================================================

def parse_namd_number(value):

    value = value.strip()

    if "*" in value:
        return np.nan

    try:
        return float(value)

    except ValueError:
        return np.nan


# ============================================================
# CHECK INPUT FILES
# ============================================================

print("=" * 80)
print("SMD BOUND - PMf UNBOUND ENERGY DIFFERENCE")
print("=" * 80)

print("\nBOUND NAMD LOG:")
print(bound_log)

print("\nUNBOUND PMf CSV:")
print(unbound_csv)

print("\nOUTPUT DIRECTORY:")
print(output_dir)


if not os.path.isfile(bound_log):

    raise FileNotFoundError(
        f"\nBOUND log not found:\n{bound_log}"
    )


if not os.path.isfile(unbound_csv):

    raise FileNotFoundError(
        f"\nUNBOUND CSV not found:\n{unbound_csv}"
    )


# ============================================================
# PARSE BOUND NAMD LOG
# ============================================================

print("\nParsing SMD bound-state NAMD log...")


energy_columns = None
bound_records = []


with open(
    bound_log,
    "r",
    encoding="utf-8",
    errors="ignore"
) as f:

    for line in f:

        stripped = line.strip()

        # ----------------------------------------------------
        # ETITLE
        # ----------------------------------------------------

        if stripped.startswith("ETITLE:"):

            detected = stripped.split()[1:]

            if detected:
                energy_columns = detected

        # ----------------------------------------------------
        # ENERGY
        # ----------------------------------------------------

        elif stripped.startswith("ENERGY:"):

            if energy_columns is None:
                continue

            parts = stripped.split()[1:]

            values = [
                parse_namd_number(x)
                for x in parts
            ]


            if len(values) < len(energy_columns):

                values += (
                    [np.nan]
                    *
                    (len(energy_columns) - len(values))
                )

            elif len(values) > len(energy_columns):

                values = values[:len(energy_columns)]


            bound_records.append(
                dict(
                    zip(
                        energy_columns,
                        values
                    )
                )
            )


if not bound_records:

    raise RuntimeError(
        "\nNo ENERGY records were extracted from the SMD log."
    )


bound = pd.DataFrame(
    bound_records
)


print(
    f"\nBOUND ENERGY records extracted: "
    f"{len(bound):,}"
)


print("\nBOUND parameters:")

for i, col in enumerate(
    bound.columns,
    start=1
):

    print(
        f"{i:02d}. {col}"
    )


# ============================================================
# ADD TIME TO BOUND DATA
# ============================================================

if "TS" not in bound.columns:

    raise RuntimeError(
        "\nTS was not found in the SMD ENERGY output."
    )


bound["TIME_FS"] = (
    bound["TS"]
    *
    DT_FS
)

bound["TIME_PS"] = (
    bound["TIME_FS"]
    /
    1000.0
)

bound["TIME_NS"] = (
    bound["TIME_FS"]
    /
    1_000_000.0
)


print(
    "\nSMD bound-state time range:"
)

print(
    f"{bound['TIME_NS'].min():.6f} - "
    f"{bound['TIME_NS'].max():.6f} ns"
)


# ============================================================
# SAVE PARSED BOUND ENERGY DATA
# ============================================================

bound.to_csv(
    output_bound_csv,
    index=False,
    sep=";",
    decimal=",",
    encoding="utf-8-sig"
)


print(
    "\n[OK] Parsed bound-state energy table saved:"
)

print(
    output_bound_csv
)


# ============================================================
# READ UNBOUND PMf DATA
# ============================================================

unbound = pd.read_csv(
    unbound_csv,
    sep=";",
    decimal=",",
    encoding="utf-8-sig"
)


print(
    f"\nUNBOUND ENERGY records loaded: "
    f"{len(unbound):,}"
)


# ============================================================
# CONVERT ALL POSSIBLE COLUMNS TO NUMERIC
# ============================================================

for col in bound.columns:

    bound[col] = pd.to_numeric(
        bound[col],
        errors="coerce"
    )


for col in unbound.columns:

    unbound[col] = pd.to_numeric(
        unbound[col],
        errors="coerce"
    )


# ============================================================
# CHECK TIME
# ============================================================

if "TIME_NS" not in unbound.columns:

    raise RuntimeError(
        "\nTIME_NS not found in PMf unbound CSV."
    )


# ============================================================
# CLEAN TIME DATA
# ============================================================

bound = (
    bound
    .dropna(
        subset=["TIME_NS"]
    )
    .sort_values(
        "TIME_NS"
    )
    .reset_index(
        drop=True
    )
)

unbound = (
    unbound
    .dropna(
        subset=["TIME_NS"]
    )
    .sort_values(
        "TIME_NS"
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# HANDLE DUPLICATE TIMESTEPS
# ============================================================

bound = (
    bound
    .groupby(
        "TIME_NS",
        as_index=False
    )
    .mean(
        numeric_only=True
    )
)

unbound = (
    unbound
    .groupby(
        "TIME_NS",
        as_index=False
    )
    .mean(
        numeric_only=True
    )
)


# ============================================================
# COMMON TIME INTERVAL
# ============================================================

start_time = max(
    bound["TIME_NS"].min(),
    unbound["TIME_NS"].min()
)

end_time = min(
    bound["TIME_NS"].max(),
    unbound["TIME_NS"].max()
)


print("\nTIME RANGES")

print(
    f"BOUND SMD: "
    f"{bound['TIME_NS'].min():.6f} - "
    f"{bound['TIME_NS'].max():.6f} ns"
)

print(
    f"UNBOUND PMf: "
    f"{unbound['TIME_NS'].min():.6f} - "
    f"{unbound['TIME_NS'].max():.6f} ns"
)

print(
    f"COMMON: "
    f"{start_time:.6f} - "
    f"{end_time:.6f} ns"
)


if end_time <= start_time:

    raise RuntimeError(
        "\nNo overlapping time interval was found."
    )


# ============================================================
# COMMON TIME GRID
# ============================================================

common_time = bound.loc[
    (
        bound["TIME_NS"] >= start_time
    )
    &
    (
        bound["TIME_NS"] <= end_time
    ),
    "TIME_NS"
].to_numpy()


if len(common_time) < 2:

    raise RuntimeError(
        "\nToo few common time points for comparison."
    )


print(
    f"\nComparison points: "
    f"{len(common_time):,}"
)


# ============================================================
# IDENTIFY COMMON PARAMETERS
# ============================================================

exclude_columns = {
    "TS",
    "TIME_FS",
    "TIME_PS",
    "TIME_NS"
}


bound_numeric = set(
    bound.select_dtypes(
        include=[np.number]
    ).columns
)


unbound_numeric = set(
    unbound.select_dtypes(
        include=[np.number]
    ).columns
)


common_parameters = sorted(
    (
        bound_numeric
        &
        unbound_numeric
    )
    -
    exclude_columns
)


if not common_parameters:

    raise RuntimeError(
        "\nNo common numeric parameters found."
    )


print("\nCOMMON PARAMETERS:")

for i, parameter in enumerate(
    common_parameters,
    start=1
):

    print(
        f"{i:02d}. {parameter}"
    )


# ============================================================
# INTERPOLATION FUNCTION
# ============================================================

def interpolate_parameter(
    dataframe,
    parameter,
    target_time
):

    temporary = dataframe[
        [
            "TIME_NS",
            parameter
        ]
    ].dropna()


    if len(temporary) < 2:

        return np.full(
            len(target_time),
            np.nan
        )


    x = temporary[
        "TIME_NS"
    ].to_numpy()

    y = temporary[
        parameter
    ].to_numpy()


    return np.interp(
        target_time,
        x,
        y
    )


# ============================================================
# CALCULATE:
#
# DELTA = SMD BOUND - PMf UNBOUND
# ============================================================

difference = pd.DataFrame({
    "TIME_NS": common_time
})


for parameter in common_parameters:

    bound_values = interpolate_parameter(
        bound,
        parameter,
        common_time
    )

    unbound_values = interpolate_parameter(
        unbound,
        parameter,
        common_time
    )


    delta_values = (
        bound_values
        -
        unbound_values
    )


    difference[
        f"{parameter}_BOUND_SMD"
    ] = bound_values

    difference[
        f"{parameter}_UNBOUND_PMF"
    ] = unbound_values

    difference[
        f"DELTA_{parameter}"
    ] = delta_values


# ============================================================
# ELECT + VDW COMBINED DIFFERENCE
# ============================================================

if (
    "ELECT" in common_parameters
    and
    "VDW" in common_parameters
):

    difference[
        "DELTA_ELECT_PLUS_VDW"
    ] = (
        difference["DELTA_ELECT"]
        +
        difference["DELTA_VDW"]
    )


# ============================================================
# SAVE COMPLETE DATA
# ============================================================

difference.to_csv(
    output_complete_csv,
    index=False,
    sep=";",
    decimal=",",
    encoding="utf-8-sig"
)


print(
    "\n[OK] Difference table saved:"
)

print(
    output_complete_csv
)


# ============================================================
# SUMMARY STATISTICS
# ============================================================

summary_rows = []


for parameter in common_parameters:

    bound_col = (
        f"{parameter}_BOUND_SMD"
    )

    unbound_col = (
        f"{parameter}_UNBOUND_PMF"
    )

    delta_col = (
        f"DELTA_{parameter}"
    )


    delta = difference[
        delta_col
    ].dropna()


    if len(delta) == 0:
        continue


    summary_rows.append({

        "PARAMETER":
            parameter,

        "N":
            len(delta),

        "BOUND_SMD_MEAN":
            difference[
                bound_col
            ].mean(),

        "BOUND_SMD_SD":
            difference[
                bound_col
            ].std(),

        "UNBOUND_PMF_MEAN":
            difference[
                unbound_col
            ].mean(),

        "UNBOUND_PMF_SD":
            difference[
                unbound_col
            ].std(),

        "DELTA_MEAN":
            delta.mean(),

        "DELTA_SD":
            delta.std(),

        "DELTA_SEM":
            delta.sem(),

        "DELTA_MEDIAN":
            delta.median(),

        "DELTA_MIN":
            delta.min(),

        "DELTA_MAX":
            delta.max()

    })


summary = pd.DataFrame(
    summary_rows
)


summary.to_csv(
    output_summary_csv,
    index=False,
    sep=";",
    decimal=",",
    encoding="utf-8-sig"
)


print(
    "\n[OK] Summary saved:"
)

print(
    output_summary_csv
)


# ============================================================
# PARAMETER CLASSES
# ============================================================

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
    "GPRESSURE",
    "PRESSAVG",
    "GPRESSAVG"
}


# ============================================================
# Y LABEL FUNCTION
# ============================================================

def get_ylabel(parameter):

    if parameter in energy_terms:

        return (
            "Δ Energy "
            "(SMD bound − PMf, kcal/mol)"
        )

    elif parameter in temperature_terms:

        return (
            "Δ Temperature "
            "(SMD bound − PMf, K)"
        )

    elif parameter in pressure_terms:

        return (
            "Δ Pressure "
            "(SMD bound − PMf, bar)"
        )

    elif parameter == "VOLUME":

        return (
            "Δ Volume "
            "(SMD bound − PMf, Å³)"
        )

    else:

        return (
            f"Δ {parameter} "
            "(SMD bound − PMf)"
        )


# ============================================================
# INDIVIDUAL DIFFERENCE PLOTS
# ============================================================

print(
    "\nCreating individual plots..."
)


for parameter in common_parameters:

    delta_col = (
        f"DELTA_{parameter}"
    )

    y = difference[
        delta_col
    ]


    if y.notna().sum() == 0:
        continue


    plt.figure(
        figsize=(6, 4),
        dpi=300
    )


    # --------------------------------------------------------
    # RAW
    # --------------------------------------------------------

    if PLOT_RAW:

        plt.plot(
            difference["TIME_NS"],
            y,
            linewidth=0.6,
            alpha=0.35,
            label="Raw Δ"
        )


    # --------------------------------------------------------
    # ROLLING MEAN
    # --------------------------------------------------------

    if ROLLING_WINDOW > 1:

        rolling = y.rolling(
            window=ROLLING_WINDOW,
            center=True,
            min_periods=1
        ).mean()


        plt.plot(
            difference["TIME_NS"],
            rolling,
            linewidth=1.2,
            label=(
                f"Rolling mean "
                f"(n={ROLLING_WINDOW})"
            )
        )


    elif not PLOT_RAW:

        plt.plot(
            difference["TIME_NS"],
            y,
            linewidth=0.8
        )


    # --------------------------------------------------------
    # ZERO
    # --------------------------------------------------------

    plt.axhline(
        0,
        linewidth=0.8,
        linestyle="--"
    )


    # --------------------------------------------------------
    # MEAN
    # --------------------------------------------------------

    mean_delta = y.mean()


    plt.axhline(
        mean_delta,
        linewidth=1.0,
        linestyle=":",
        label=(
            f"Mean Δ = "
            f"{mean_delta:.3f}"
        )
    )


    plt.xlabel(
        "Time (ns)"
    )

    plt.ylabel(
        get_ylabel(parameter)
    )

    plt.title(
        f"Δ{parameter}\n"
        "SMD PMf + MMV675968 − PMf"
    )

    plt.legend(
        fontsize=8,
        frameon=False
    )

    plt.tight_layout()


    safe_parameter = re.sub(
        r"[^A-Za-z0-9_-]",
        "_",
        parameter
    )


    individual_png = os.path.join(
        output_dir,
        f"DELTA_{safe_parameter}.png"
    )


    plt.savefig(
        individual_png,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


print(
    "[OK] Individual plots completed."
)


# ============================================================
# COMBINED ENERGY COMPONENTS
# ============================================================

component_terms = [
    "BOND",
    "ANGLE",
    "DIHED",
    "IMPRP",
    "ELECT",
    "VDW",
    "BOUNDARY",
    "MISC"
]


available_components = [
    parameter
    for parameter in component_terms
    if parameter in common_parameters
]


if available_components:

    plt.figure(
        figsize=(8, 5),
        dpi=300
    )


    for parameter in available_components:

        y = difference[
            f"DELTA_{parameter}"
        ]


        if ROLLING_WINDOW > 1:

            y = y.rolling(
                window=ROLLING_WINDOW,
                center=True,
                min_periods=1
            ).mean()


        plt.plot(
            difference["TIME_NS"],
            y,
            linewidth=1.0,
            label=parameter
        )


    plt.axhline(
        0,
        linewidth=0.8,
        linestyle="--"
    )


    plt.xlabel(
        "Time (ns)"
    )

    plt.ylabel(
        "Δ Energy "
        "(SMD bound − PMf, kcal/mol)"
    )

    plt.title(
        "Energy Components\n"
        "SMD PMf + MMV675968 − PMf"
    )

    plt.legend(
        fontsize=8,
        frameon=False,
        ncol=2
    )

    plt.tight_layout()


    plt.savefig(
        output_components_png,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# GLOBAL ENERGIES
# ============================================================

global_terms = [
    "TOTAL",
    "POTENTIAL",
    "KINETIC",
    "TOTAL2",
    "TOTAL3"
]


available_global = [
    parameter
    for parameter in global_terms
    if parameter in common_parameters
]


if available_global:

    plt.figure(
        figsize=(8, 5),
        dpi=300
    )


    for parameter in available_global:

        y = difference[
            f"DELTA_{parameter}"
        ]


        if ROLLING_WINDOW > 1:

            y = y.rolling(
                window=ROLLING_WINDOW,
                center=True,
                min_periods=1
            ).mean()


        plt.plot(
            difference["TIME_NS"],
            y,
            linewidth=1.0,
            label=parameter
        )


    plt.axhline(
        0,
        linewidth=0.8,
        linestyle="--"
    )


    plt.xlabel(
        "Time (ns)"
    )

    plt.ylabel(
        "Δ Energy "
        "(SMD bound − PMf, kcal/mol)"
    )

    plt.title(
        "Global Energies\n"
        "SMD PMf + MMV675968 − PMf"
    )

    plt.legend(
        fontsize=8,
        frameon=False
    )

    plt.tight_layout()


    plt.savefig(
        output_global_png,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# ELECTROSTATIC + VDW
# ============================================================

if (
    "ELECT" in common_parameters
    or
    "VDW" in common_parameters
):

    plt.figure(
        figsize=(7, 5),
        dpi=300
    )


    # ELECT
    if "ELECT" in common_parameters:

        y = difference[
            "DELTA_ELECT"
        ]

        if ROLLING_WINDOW > 1:

            y = y.rolling(
                window=ROLLING_WINDOW,
                center=True,
                min_periods=1
            ).mean()


        plt.plot(
            difference["TIME_NS"],
            y,
            linewidth=1.0,
            label="ΔELECT"
        )


    # VDW
    if "VDW" in common_parameters:

        y = difference[
            "DELTA_VDW"
        ]

        if ROLLING_WINDOW > 1:

            y = y.rolling(
                window=ROLLING_WINDOW,
                center=True,
                min_periods=1
            ).mean()


        plt.plot(
            difference["TIME_NS"],
            y,
            linewidth=1.0,
            label="ΔVDW"
        )


    # ELECT + VDW
    if (
        "DELTA_ELECT_PLUS_VDW"
        in difference.columns
    ):

        y = difference[
            "DELTA_ELECT_PLUS_VDW"
        ]

        if ROLLING_WINDOW > 1:

            y = y.rolling(
                window=ROLLING_WINDOW,
                center=True,
                min_periods=1
            ).mean()


        plt.plot(
            difference["TIME_NS"],
            y,
            linewidth=1.4,
            label="Δ(ELECT + VDW)"
        )


    plt.axhline(
        0,
        linewidth=0.8,
        linestyle="--"
    )


    plt.xlabel(
        "Time (ns)"
    )

    plt.ylabel(
        "Δ Energy "
        "(SMD bound − PMf, kcal/mol)"
    )

    plt.title(
        "Electrostatic and van der Waals Differences\n"
        "SMD PMf + MMV675968 − PMf"
    )

    plt.legend(
        fontsize=8,
        frameon=False
    )

    plt.tight_layout()


    plt.savefig(
        output_nonbonded_png,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# BAR PLOT OF MEAN ENERGY DIFFERENCES
# ============================================================

energy_summary = summary[
    summary[
        "PARAMETER"
    ].isin(
        energy_terms
    )
].copy()


if not energy_summary.empty:

    desired_order = [
        "BOND",
        "ANGLE",
        "DIHED",
        "IMPRP",
        "ELECT",
        "VDW",
        "BOUNDARY",
        "MISC",
        "POTENTIAL",
        "KINETIC",
        "TOTAL",
        "TOTAL2",
        "TOTAL3"
    ]


    energy_summary[
        "ORDER"
    ] = energy_summary[
        "PARAMETER"
    ].apply(
        lambda x:
        (
            desired_order.index(x)
            if x in desired_order
            else 999
        )
    )


    energy_summary = (
        energy_summary
        .sort_values(
            "ORDER"
        )
    )


    plt.figure(
        figsize=(9, 5),
        dpi=300
    )


    plt.bar(
        energy_summary[
            "PARAMETER"
        ],
        energy_summary[
            "DELTA_MEAN"
        ],
        yerr=energy_summary[
            "DELTA_SD"
        ],
        capsize=3
    )


    plt.axhline(
        0,
        linewidth=0.8,
        linestyle="--"
    )


    plt.xlabel(
        "Energy parameter"
    )

    plt.ylabel(
        "Mean Δ Energy "
        "(SMD bound − PMf, kcal/mol)"
    )

    plt.title(
        "Mean Energy Differences"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()


    plt.savefig(
        output_mean_png,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# PRINT MAIN RESULTS
# ============================================================

print("\n" + "=" * 80)
print("MEAN ENERGY DIFFERENCES")
print("DELTA = SMD BOUND - PMf UNBOUND")
print("=" * 80)


important_terms = [
    "BOND",
    "ANGLE",
    "DIHED",
    "IMPRP",
    "ELECT",
    "VDW",
    "POTENTIAL",
    "KINETIC",
    "TOTAL"
]


for parameter in important_terms:

    row = summary[
        summary[
            "PARAMETER"
        ]
        ==
        parameter
    ]


    if row.empty:
        continue


    mean_delta = row[
        "DELTA_MEAN"
    ].iloc[0]

    sd_delta = row[
        "DELTA_SD"
    ].iloc[0]


    print(
        f"{parameter:12s} "
        f"{mean_delta:15.4f} "
        f"± {sd_delta:.4f} kcal/mol"
    )


# ============================================================
# NONBONDED DIFFERENCE
# ============================================================

if (
    "DELTA_ELECT_PLUS_VDW"
    in difference.columns
):

    nb = difference[
        "DELTA_ELECT_PLUS_VDW"
    ].dropna()


    print(
        "\nCombined nonbonded difference:"
    )

    print(
        "Δ(ELECT + VDW) = "
        f"{nb.mean():.4f} "
        f"± {nb.std():.4f} kcal/mol"
    )


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)


print(
    f"""
Definition:

    DELTA = SMD BOUND - PMf UNBOUND


BOUND:

    PMf + MMV675968
    60 A / 1.5 ns SMD

Input log:

{bound_log}


UNBOUND:

    PMf alone

Input CSV:

{unbound_csv}


Common analyzed time interval:

    {start_time:.6f} - {end_time:.6f} ns


Comparison points:

    {len(common_time):,}


Shared parameters:

    {len(common_parameters)}


ALL RESULTS SAVED TO:

{output_dir}


Main output files:

MMV675968_SMD_bound_energy_data.csv

MMV675968_SMD_bound_minus_unbound_energy_difference.csv

MMV675968_SMD_bound_minus_unbound_summary.csv

MMV675968_SMD_delta_energy_components.png

MMV675968_SMD_delta_global_energies.png

MMV675968_SMD_delta_electrostatic_vdw.png

MMV675968_SMD_mean_energy_differences.png

DELTA_BOND.png
DELTA_ANGLE.png
DELTA_DIHED.png
DELTA_IMPRP.png
DELTA_ELECT.png
DELTA_VDW.png
DELTA_POTENTIAL.png
DELTA_TOTAL.png
etc.
"""
)