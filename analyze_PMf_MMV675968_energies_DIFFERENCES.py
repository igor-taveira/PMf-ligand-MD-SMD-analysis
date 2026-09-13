import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# NAMD ENERGY DIFFERENCE ANALYSIS
#
# Definition:
#
#   DELTA = BOUND - UNBOUND
#
#   BOUND   = PMf + MMV675968
#   UNBOUND = PMf alone
# ============================================================


# ============================================================
# PATHS
# ============================================================

project_dir = (
    r"D:\fcm-uerj\artigos-e-producao"
    r"\2027_candida-MMV687807-675968_daniel"
)


# ------------------------------------------------------------
# BOUND SYSTEM
# PMf + MMV675968
# ------------------------------------------------------------

bound_dir = os.path.join(
    project_dir,
    "PMf_MMV675968_SMD",
    "NAMD_PMf_MMV675968_staged"
)

bound_csv = os.path.join(
    bound_dir,
    "MMV675968_energy_data.csv"
)


# ------------------------------------------------------------
# UNBOUND SYSTEM
# PMf alone
# ------------------------------------------------------------

unbound_dir = os.path.join(
    project_dir,
    "PMf",
    "NAMD_PMf_staged"
)

unbound_csv = os.path.join(
    unbound_dir,
    "PMf_energy_data.csv"
)


# ------------------------------------------------------------
# OUTPUT SUBFOLDER
# ------------------------------------------------------------

output_dir = os.path.join(
    bound_dir,
    "energy_difference_bound_minus_unbound"
)

os.makedirs(
    output_dir,
    exist_ok=True
)


# ============================================================
# OUTPUT FILES
# ============================================================

output_complete_csv = os.path.join(
    output_dir,
    "MMV675968_bound_minus_unbound_energy_difference.csv"
)

output_summary_csv = os.path.join(
    output_dir,
    "MMV675968_bound_minus_unbound_summary.csv"
)

output_components_png = os.path.join(
    output_dir,
    "MMV675968_delta_energy_components.png"
)

output_global_png = os.path.join(
    output_dir,
    "MMV675968_delta_global_energies.png"
)

output_nonbonded_png = os.path.join(
    output_dir,
    "MMV675968_delta_electrostatic_vdw.png"
)

output_mean_png = os.path.join(
    output_dir,
    "MMV675968_mean_energy_differences.png"
)


# ============================================================
# SETTINGS
# ============================================================

# Number of points used for rolling average.
# Change to 1 if you want only raw data.
ROLLING_WINDOW = 50

SAVE_RAW_AND_SMOOTHED = True


# ============================================================
# INITIAL INFORMATION
# ============================================================

print("=" * 80)
print("NAMD ENERGY DIFFERENCE ANALYSIS")
print("BOUND - UNBOUND")
print("=" * 80)

print("\nBOUND:")
print(bound_csv)

print("\nUNBOUND:")
print(unbound_csv)

print("\nOUTPUT:")
print(output_dir)


# ============================================================
# CHECK INPUT FILES
# ============================================================

if not os.path.isfile(bound_csv):

    raise FileNotFoundError(
        "\nBOUND CSV was not found:\n"
        f"{bound_csv}"
    )


if not os.path.isfile(unbound_csv):

    raise FileNotFoundError(
        "\nUNBOUND CSV was not found:\n"
        f"{unbound_csv}"
    )


# ============================================================
# READ CSV FILES
#
# Previous energy scripts used:
# separator = ;
# decimal   = ,
# ============================================================

bound = pd.read_csv(
    bound_csv,
    sep=";",
    decimal=",",
    encoding="utf-8-sig"
)

unbound = pd.read_csv(
    unbound_csv,
    sep=";",
    decimal=",",
    encoding="utf-8-sig"
)


print("\nFiles loaded successfully.")

print(
    f"BOUND rows:   {len(bound):,}"
)

print(
    f"UNBOUND rows: {len(unbound):,}"
)


# ============================================================
# CONVERT COLUMNS TO NUMERIC
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
# CHECK TIME COLUMN
# ============================================================

if "TIME_NS" not in bound.columns:

    raise RuntimeError(
        "\nTIME_NS was not found in the BOUND CSV."
    )


if "TIME_NS" not in unbound.columns:

    raise RuntimeError(
        "\nTIME_NS was not found in the UNBOUND CSV."
    )


# ============================================================
# CLEAN TIME DATA
# ============================================================

bound = bound.dropna(
    subset=["TIME_NS"]
).copy()

unbound = unbound.dropna(
    subset=["TIME_NS"]
).copy()


bound = bound.sort_values(
    "TIME_NS"
).reset_index(drop=True)

unbound = unbound.sort_values(
    "TIME_NS"
).reset_index(drop=True)


# ============================================================
# HANDLE DUPLICATE TIME POINTS
#
# Restarted NAMD simulations can contain repeated timesteps.
# Mean is taken for duplicate TIME_NS entries.
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
# DETERMINE COMMON TIME INTERVAL
# ============================================================

start_time = max(
    bound["TIME_NS"].min(),
    unbound["TIME_NS"].min()
)

end_time = min(
    bound["TIME_NS"].max(),
    unbound["TIME_NS"].max()
)


print("\nTime ranges:")

print(
    f"BOUND:   "
    f"{bound['TIME_NS'].min():.6f} - "
    f"{bound['TIME_NS'].max():.6f} ns"
)

print(
    f"UNBOUND: "
    f"{unbound['TIME_NS'].min():.6f} - "
    f"{unbound['TIME_NS'].max():.6f} ns"
)

print(
    f"COMMON:  "
    f"{start_time:.6f} - "
    f"{end_time:.6f} ns"
)


if end_time <= start_time:

    raise RuntimeError(
        "\nThe simulations have no overlapping time interval."
    )


# ============================================================
# USE BOUND TIME POINTS INSIDE COMMON INTERVAL
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
        "\nNot enough common time points."
    )


print(
    f"\nComparison time points: {len(common_time):,}"
)


# ============================================================
# IDENTIFY COMMON NUMERIC PARAMETERS
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
        "\nNo common numerical parameters were found."
    )


print("\nCommon parameters detected:")

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
        ["TIME_NS", parameter]
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
# CALCULATE DIFFERENCES
#
# DELTA = BOUND - UNBOUND
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
        f"{parameter}_BOUND"
    ] = bound_values

    difference[
        f"{parameter}_UNBOUND"
    ] = unbound_values

    difference[
        f"DELTA_{parameter}"
    ] = delta_values


# ============================================================
# SAVE COMPLETE DIFFERENCE TABLE
# ============================================================

difference.to_csv(
    output_complete_csv,
    index=False,
    sep=";",
    decimal=",",
    encoding="utf-8-sig"
)


print(
    "\n[OK] Complete difference table saved:"
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
        f"{parameter}_BOUND"
    )

    unbound_col = (
        f"{parameter}_UNBOUND"
    )

    delta_col = (
        f"DELTA_{parameter}"
    )


    valid = difference[
        delta_col
    ].dropna()


    if len(valid) == 0:
        continue


    summary_rows.append({

        "PARAMETER":
            parameter,

        "N":
            len(valid),

        "BOUND_MEAN":
            difference[
                bound_col
            ].mean(),

        "BOUND_SD":
            difference[
                bound_col
            ].std(),

        "UNBOUND_MEAN":
            difference[
                unbound_col
            ].mean(),

        "UNBOUND_SD":
            difference[
                unbound_col
            ].std(),

        "DELTA_MEAN":
            valid.mean(),

        "DELTA_SD":
            valid.std(),

        "DELTA_SEM":
            valid.sem(),

        "DELTA_MEDIAN":
            valid.median(),

        "DELTA_MIN":
            valid.min(),

        "DELTA_MAX":
            valid.max()

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
# PARAMETER GROUPS
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
# Y-AXIS LABEL FUNCTION
# ============================================================

def get_ylabel(parameter):

    if parameter in energy_terms:

        return (
            "Δ Energy "
            "(bound − unbound, kcal/mol)"
        )

    elif parameter in temperature_terms:

        return (
            "Δ Temperature "
            "(bound − unbound, K)"
        )

    elif parameter in pressure_terms:

        return (
            "Δ Pressure "
            "(bound − unbound, bar)"
        )

    elif parameter == "VOLUME":

        return (
            "Δ Volume "
            "(bound − unbound, Å³)"
        )

    else:

        return (
            f"Δ {parameter} "
            "(bound − unbound)"
        )


# ============================================================
# INDIVIDUAL PARAMETER PLOTS
# ============================================================

print(
    "\nCreating individual Δ plots..."
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
    # RAW DATA
    # --------------------------------------------------------

    if SAVE_RAW_AND_SMOOTHED:

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


    elif not SAVE_RAW_AND_SMOOTHED:

        plt.plot(
            difference["TIME_NS"],
            y,
            linewidth=0.8
        )


    # --------------------------------------------------------
    # ZERO REFERENCE
    # --------------------------------------------------------

    plt.axhline(
        0,
        linewidth=0.8,
        linestyle="--"
    )


    # --------------------------------------------------------
    # MEAN DELTA
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
        f"Δ{parameter}: "
        "PMf + MMV675968 − PMf"
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
        "(bound − unbound, kcal/mol)"
    )

    plt.title(
        "Energy Components\n"
        "PMf + MMV675968 − PMf"
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
        "(bound − unbound, kcal/mol)"
    )

    plt.title(
        "Global Energies\n"
        "PMf + MMV675968 − PMf"
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
# ELECTROSTATIC + VAN DER WAALS
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


    if "ELECT" in common_parameters:

        delta_elect = difference[
            "DELTA_ELECT"
        ]

        if ROLLING_WINDOW > 1:

            delta_elect_plot = (
                delta_elect
                .rolling(
                    window=ROLLING_WINDOW,
                    center=True,
                    min_periods=1
                )
                .mean()
            )

        else:

            delta_elect_plot = (
                delta_elect
            )


        plt.plot(
            difference["TIME_NS"],
            delta_elect_plot,
            linewidth=1.0,
            label="ΔELECT"
        )


    if "VDW" in common_parameters:

        delta_vdw = difference[
            "DELTA_VDW"
        ]

        if ROLLING_WINDOW > 1:

            delta_vdw_plot = (
                delta_vdw
                .rolling(
                    window=ROLLING_WINDOW,
                    center=True,
                    min_periods=1
                )
                .mean()
            )

        else:

            delta_vdw_plot = (
                delta_vdw
            )


        plt.plot(
            difference["TIME_NS"],
            delta_vdw_plot,
            linewidth=1.0,
            label="ΔVDW"
        )


    # --------------------------------------------------------
    # ELECT + VDW
    # --------------------------------------------------------

    if (
        "ELECT" in common_parameters
        and
        "VDW" in common_parameters
    ):

        delta_nonbonded = (
            difference["DELTA_ELECT"]
            +
            difference["DELTA_VDW"]
        )


        difference[
            "DELTA_ELECT_PLUS_VDW"
        ] = delta_nonbonded


        if ROLLING_WINDOW > 1:

            delta_nonbonded_plot = (
                delta_nonbonded
                .rolling(
                    window=ROLLING_WINDOW,
                    center=True,
                    min_periods=1
                )
                .mean()
            )

        else:

            delta_nonbonded_plot = (
                delta_nonbonded
            )


        plt.plot(
            difference["TIME_NS"],
            delta_nonbonded_plot,
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
        "(bound − unbound, kcal/mol)"
    )

    plt.title(
        "Electrostatic and van der Waals Differences\n"
        "PMf + MMV675968 − PMf"
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
# SAVE AGAIN IF ELECT+VDW COLUMN WAS ADDED
# ============================================================

difference.to_csv(
    output_complete_csv,
    index=False,
    sep=";",
    decimal=",",
    encoding="utf-8-sig"
)


# ============================================================
# BAR PLOT OF MEAN ENERGY DIFFERENCES
# ============================================================

energy_summary = summary[
    summary["PARAMETER"].isin(
        energy_terms
    )
].copy()


if not energy_summary.empty:

    # Preserve a meaningful order
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
        .sort_values("ORDER")
    )


    plt.figure(
        figsize=(9, 5),
        dpi=300
    )


    plt.bar(
        energy_summary["PARAMETER"],
        energy_summary["DELTA_MEAN"],
        yerr=energy_summary["DELTA_SD"],
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
        "(bound − unbound, kcal/mol)"
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
# PRINT MAIN ENERGY RESULTS
# ============================================================

print("\n" + "=" * 80)
print("MEAN ENERGY DIFFERENCES")
print("Δ = BOUND - UNBOUND")
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
        summary["PARAMETER"]
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
# NONBONDED MEAN
# ============================================================

if (
    "DELTA_ELECT_PLUS_VDW"
    in difference.columns
):

    nb = difference[
        "DELTA_ELECT_PLUS_VDW"
    ].dropna()


    print("\nCombined nonbonded difference:")

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

    DELTA = BOUND - UNBOUND

    BOUND:
    PMf + MMV675968

    UNBOUND:
    PMf alone


Common time interval:

    {start_time:.6f} - {end_time:.6f} ns


Comparison points:

    {len(common_time):,}


Shared parameters:

    {len(common_parameters)}


All results were saved to:

{output_dir}


Main files:

MMV675968_bound_minus_unbound_energy_difference.csv

MMV675968_bound_minus_unbound_summary.csv

MMV675968_delta_energy_components.png

MMV675968_delta_global_energies.png

MMV675968_delta_electrostatic_vdw.png

MMV675968_mean_energy_differences.png

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