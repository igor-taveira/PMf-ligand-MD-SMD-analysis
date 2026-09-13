import os
import re
import itertools
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.stattools import durbin_watson


# ============================================================
# PATHS
# ============================================================

# Parent folder containing the NAMD energy log
base_dir = os.environ.get(
    "MMV675968_BASE_DIR",
    (
        r"D:\fcm-uerj\artigos-e-producao"
        r"\2027_candida-MMV687807-675968_daniel"
        r"\PMf_MMV675968_SMD"
    )
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

# Detailed temporal inference in spreadsheet-friendly and human-readable forms
output_temporal_statistics_csv = os.path.join(
    output_dir,
    "MMV675968_energy_temporal_statistics.csv"
)

output_temporal_statistics_txt = os.path.join(
    output_dir,
    "MMV675968_energy_temporal_statistics.txt"
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
# FUNCTIONS FOR TIME-SERIES-AWARE STATISTICAL INFERENCE
# ============================================================

def autocorrelation_fft(values):
    """Return the normalized autocorrelation function using an FFT."""

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size < 3:
        return np.array([1.0])

    centered = values - np.mean(values)
    variance_sum = np.dot(centered, centered)

    if variance_sum <= 0:
        return np.array([1.0])

    fft_size = 1 << (2 * values.size - 1).bit_length()
    transformed = np.fft.rfft(centered, n=fft_size)
    autocovariance = np.fft.irfft(
        transformed * np.conjugate(transformed),
        n=fft_size
    )[:values.size]

    # Unbiased normalization for the decreasing number of pairs at each lag.
    autocovariance /= np.arange(values.size, 0, -1)
    return autocovariance / autocovariance[0]


def estimate_statistical_inefficiency(values):
    """
    Estimate statistical inefficiency g = 1 + 2*sum(rho_k).

    The sum follows the initial-positive-sequence rule and is capped before
    very long, noisy lags dominate the estimate. The effective sample size is
    approximately N/g, and ceil(g) is used as the automatic block length.
    """

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n_values = values.size

    if n_values < 3 or np.nanstd(values, ddof=1) == 0:
        return 1.0, float(n_values), 0.0

    acf = autocorrelation_fft(values)
    max_lag = min(n_values // 2, 10000)
    positive_sum = 0.0

    for lag in range(1, max_lag + 1):
        rho = acf[lag]
        if not np.isfinite(rho) or rho <= 0:
            break
        positive_sum += rho

    statistical_inefficiency = max(1.0, 1.0 + 2.0 * positive_sum)
    effective_n = max(1.0, min(float(n_values), n_values / statistical_inefficiency))
    acf_lag1 = float(acf[1]) if acf.size > 1 else np.nan

    return statistical_inefficiency, effective_n, acf_lag1


def make_nonoverlapping_blocks(time_values, metric_values, block_size):
    """Calculate time and metric means for complete, non-overlapping blocks."""

    time_values = np.asarray(time_values, dtype=float)
    metric_values = np.asarray(metric_values, dtype=float)
    block_size = max(1, int(block_size))
    n_blocks = metric_values.size // block_size

    if n_blocks < 1:
        return np.array([]), np.array([])

    usable = n_blocks * block_size
    block_time = time_values[:usable].reshape(n_blocks, block_size).mean(axis=1)
    block_metric = metric_values[:usable].reshape(n_blocks, block_size).mean(axis=1)

    return block_time, block_metric


def safe_shapiro(residuals):
    """Shapiro-Wilk test with a deterministic cap at 5,000 residuals."""

    residuals = np.asarray(residuals, dtype=float)
    residuals = residuals[np.isfinite(residuals)]

    if residuals.size < 3 or np.std(residuals, ddof=1) == 0:
        return np.nan, np.nan, int(residuals.size)

    if residuals.size > 5000:
        indices = np.linspace(0, residuals.size - 1, 5000, dtype=int)
        tested = residuals[indices]
    else:
        tested = residuals

    statistic, p_value = stats.shapiro(tested)
    return float(statistic), float(p_value), int(tested.size)


def spearman_with_small_sample_exact_p(x_values, y_values):
    """Spearman rho with an exact two-sided permutation p-value for N <= 8."""

    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    n_values = y_values.size

    if (
        n_values < 3
        or np.std(x_values, ddof=1) == 0
        or np.std(y_values, ddof=1) == 0
    ):
        return np.nan, np.nan, "Not calculated"

    observed_rho = float(stats.spearmanr(x_values, y_values).statistic)

    if n_values <= 8:
        exceedances = 0
        total_permutations = math.factorial(n_values)
        observed_absolute = abs(observed_rho)

        for permuted_y in itertools.permutations(y_values.tolist()):
            permuted_rho = stats.spearmanr(x_values, permuted_y).statistic
            if abs(permuted_rho) >= observed_absolute - 1e-12:
                exceedances += 1

        exact_p = exceedances / total_permutations
        return observed_rho, float(exact_p), "Exact two-sided permutation"

    spearman_result = stats.spearmanr(x_values, y_values)
    return (
        observed_rho,
        float(spearman_result.pvalue),
        "Asymptotic two-sided"
    )


def analyze_temporal_parameter(time_values, metric_values, metric_name):
    """Analyze one NAMD parameter while accounting for temporal dependence."""

    valid = np.isfinite(time_values) & np.isfinite(metric_values)
    time_clean = np.asarray(time_values[valid], dtype=float)
    metric_clean = np.asarray(metric_values[valid], dtype=float)
    n_raw = metric_clean.size

    result = {
        "Parameter": metric_name,
        "N_raw": n_raw
    }

    if n_raw < 3 or np.unique(time_clean).size < 2:
        result["Analysis_status"] = "Insufficient valid observations"
        return result

    order = np.argsort(time_clean)
    time_clean = time_clean[order]
    metric_clean = metric_clean[order]

    g_value, effective_n, metric_acf1 = estimate_statistical_inefficiency(
        metric_clean
    )
    block_size = max(1, int(np.ceil(g_value)))
    block_time, block_metric = make_nonoverlapping_blocks(
        time_clean,
        metric_clean,
        block_size
    )

    block_note = "Autocorrelation-derived block size"
    if block_metric.size < 3:
        block_note = (
            "Fewer than 3 autocorrelation-derived blocks; block inference "
            "not calculated"
        )

    # Raw OLS point estimates with HAC/Newey-West standard errors. R-squared
    # describes the raw linear fit; inference uses autocorrelation-robust SEs.
    raw_design = sm.add_constant(time_clean)
    raw_ols = sm.OLS(metric_clean, raw_design).fit()
    hac_maxlags = max(1, min(block_size, n_raw // 4))
    raw_hac = raw_ols.get_robustcov_results(
        cov_type="HAC",
        maxlags=hac_maxlags,
        use_correction=True
    )

    raw_residuals = raw_ols.resid
    raw_dw = float(durbin_watson(raw_residuals))
    _, _, residual_acf1 = estimate_statistical_inefficiency(raw_residuals)

    ljung_box_lag = max(1, min(10, n_raw // 5))
    try:
        ljung_box_p = float(
            acorr_ljungbox(
                raw_residuals,
                lags=[ljung_box_lag],
                return_df=True
            )["lb_pvalue"].iloc[0]
        )
    except Exception:
        ljung_box_p = np.nan

    # Block-averaged regression and Spearman correlation provide a more
    # conservative analysis because blocks are approximately decorrelated.
    if block_metric.size >= 3 and np.unique(block_time).size >= 2:
        block_design = sm.add_constant(block_time)
        block_ols = sm.OLS(block_metric, block_design).fit()
        block_ci = block_ols.conf_int(alpha=0.05)
        block_residuals = block_ols.resid
        shapiro_w, shapiro_p, shapiro_n = safe_shapiro(block_residuals)
        jarque_bera = stats.jarque_bera(block_residuals)
        spearman_rho, spearman_p, spearman_method = (
            spearman_with_small_sample_exact_p(block_time, block_metric)
        )

        block_slope = float(block_ols.params[1])
        block_slope_se = float(block_ols.bse[1])
        block_slope_t = float(block_ols.tvalues[1])
        block_slope_p = float(block_ols.pvalues[1])
        block_ci_low = float(block_ci[1, 0])
        block_ci_high = float(block_ci[1, 1])
        block_intercept = float(block_ols.params[0])
        block_r2 = float(block_ols.rsquared)
        block_r2_adjusted = float(block_ols.rsquared_adj)
        block_rmse = float(np.sqrt(np.mean(np.square(block_residuals))))
        block_aic = float(block_ols.aic)
        block_bic = float(block_ols.bic)
        block_residual_dw = float(durbin_watson(block_residuals))
        jarque_bera_stat = float(jarque_bera.statistic)
        jarque_bera_p = float(jarque_bera.pvalue)
    else:
        shapiro_w = shapiro_p = np.nan
        shapiro_n = int(block_metric.size)
        spearman_rho = spearman_p = np.nan
        spearman_method = "Not calculated"
        block_slope = block_slope_se = block_slope_t = block_slope_p = np.nan
        block_ci_low = block_ci_high = block_intercept = np.nan
        block_r2 = block_r2_adjusted = block_residual_dw = np.nan
        block_rmse = block_aic = block_bic = np.nan
        jarque_bera_stat = jarque_bera_p = np.nan

    result.update({
        "Analysis_status": "OK",
        "Mean": float(np.mean(metric_clean)),
        "SD": float(np.std(metric_clean, ddof=1)),
        "Median": float(np.median(metric_clean)),
        "IQR": float(stats.iqr(metric_clean, nan_policy="omit")),
        "Minimum": float(np.min(metric_clean)),
        "Maximum": float(np.max(metric_clean)),
        "Metric_ACF_lag1": metric_acf1,
        "Statistical_inefficiency_g": g_value,
        "Effective_sample_size": effective_n,
        "Automatic_block_size_points": block_size,
        "N_complete_blocks": int(block_metric.size),
        "Block_inference_available": bool(block_metric.size >= 3),
        "Block_definition_note": block_note,
        "Raw_OLS_intercept": float(raw_ols.params[0]),
        "Raw_OLS_slope_per_time_unit": float(raw_ols.params[1]),
        "Raw_OLS_R2": float(raw_ols.rsquared),
        "Raw_OLS_adjusted_R2": float(raw_ols.rsquared_adj),
        "Raw_OLS_RMSE": float(np.sqrt(np.mean(np.square(raw_residuals)))),
        "Raw_OLS_AIC": float(raw_ols.aic),
        "Raw_OLS_BIC": float(raw_ols.bic),
        "HAC_maxlags": hac_maxlags,
        "HAC_slope_SE": float(raw_hac.bse[1]),
        "HAC_slope_t": float(raw_hac.tvalues[1]),
        "HAC_slope_p": float(raw_hac.pvalues[1]),
        "HAC_slope_CI95_low": float(raw_hac.conf_int(alpha=0.05)[1, 0]),
        "HAC_slope_CI95_high": float(raw_hac.conf_int(alpha=0.05)[1, 1]),
        "Raw_residual_ACF_lag1": residual_acf1,
        "Raw_residual_Durbin_Watson": raw_dw,
        "Raw_residual_Ljung_Box_lag": ljung_box_lag,
        "Raw_residual_Ljung_Box_p": ljung_box_p,
        "Block_OLS_intercept": block_intercept,
        "Block_OLS_slope_per_time_unit": block_slope,
        "Block_OLS_slope_SE": block_slope_se,
        "Block_OLS_slope_t": block_slope_t,
        "Block_OLS_slope_p": block_slope_p,
        "Block_OLS_slope_CI95_low": block_ci_low,
        "Block_OLS_slope_CI95_high": block_ci_high,
        "Block_OLS_R2": block_r2,
        "Block_OLS_adjusted_R2": block_r2_adjusted,
        "Block_OLS_RMSE": block_rmse,
        "Block_OLS_AIC": block_aic,
        "Block_OLS_BIC": block_bic,
        "Block_residual_Durbin_Watson": block_residual_dw,
        "Block_residual_Shapiro_W": shapiro_w,
        "Block_residual_Shapiro_p": shapiro_p,
        "Block_residual_Shapiro_N": shapiro_n,
        "Block_residual_Jarque_Bera": jarque_bera_stat,
        "Block_residual_Jarque_Bera_p": jarque_bera_p,
        "Block_Spearman_rho_time": float(spearman_rho),
        "Block_Spearman_p": float(spearman_p),
        "Block_Spearman_p_method": spearman_method
    })

    return result


def format_number(value, digits=6):
    """Format numeric report values without failing on missing results."""

    if value is None or not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}g}"


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
# TEMPORAL INFERENCE AND SPEARMAN CORRELATION
# ============================================================

print("\nCalculating time-series-aware statistical inference...")

if "TIME_NS" in df.columns:
    statistical_time = df["TIME_NS"].to_numpy(dtype=float)
    statistical_time_name = "TIME_NS"
    statistical_time_unit = "ns"
elif "TS" in df.columns:
    statistical_time = df["TS"].to_numpy(dtype=float)
    statistical_time_name = "TS"
    statistical_time_unit = "timestep"
else:
    statistical_time = np.arange(len(df), dtype=float)
    statistical_time_name = "FRAME"
    statistical_time_unit = "frame"


temporal_results = []

for col in plot_columns:
    temporal_results.append(
        analyze_temporal_parameter(
            statistical_time,
            df[col].to_numpy(dtype=float),
            col
        )
    )


temporal_statistics = pd.DataFrame(temporal_results)
temporal_statistics.insert(1, "Time_variable", statistical_time_name)
temporal_statistics.insert(2, "Time_unit", statistical_time_unit)


def add_fdr_column(table, p_column, output_column):
    """Add Benjamini-Hochberg-adjusted p-values without altering missing rows."""

    adjusted = np.full(len(table), np.nan, dtype=float)
    if p_column in table.columns:
        p_values = pd.to_numeric(table[p_column], errors="coerce").to_numpy()
        valid = np.isfinite(p_values)
        if valid.any():
            adjusted[valid] = multipletests(
                p_values[valid],
                alpha=0.05,
                method="fdr_bh"
            )[1]
    table[output_column] = adjusted


add_fdr_column(
    temporal_statistics,
    "HAC_slope_p",
    "HAC_slope_p_FDR_BH"
)
add_fdr_column(
    temporal_statistics,
    "Block_OLS_slope_p",
    "Block_OLS_slope_p_FDR_BH"
)
add_fdr_column(
    temporal_statistics,
    "Block_Spearman_p",
    "Block_Spearman_p_FDR_BH"
)


temporal_statistics.to_csv(
    output_temporal_statistics_csv,
    index=False,
    sep=";",
    decimal=",",
    encoding="utf-8-sig"
)


with open(
    output_temporal_statistics_txt,
    "w",
    encoding="utf-8"
) as report:

    report.write("NAMD ENERGY TEMPORAL STATISTICAL REPORT - MMV675968\n")
    report.write("=" * 78 + "\n\n")
    report.write(f"Input log: {log_file}\n")
    report.write(f"Time variable: {statistical_time_name}\n")
    report.write(f"Slope unit: parameter unit per {statistical_time_unit}\n")
    report.write(f"Raw energy records: {len(df):,}\n")
    report.write(f"Parameters analyzed: {len(plot_columns)}\n\n")

    report.write("STATISTICAL STRATEGY\n")
    report.write("-" * 78 + "\n")
    report.write(
        "1. Raw OLS regression estimates the linear temporal trend. Its slope "
        "inference uses HAC/Newey-West standard errors to reduce bias from "
        "heteroskedasticity and serial correlation.\n"
    )
    report.write(
        "2. Statistical inefficiency g is estimated from the initial positive "
        "autocorrelation sequence. Effective N is approximately N/g.\n"
    )
    report.write(
        "3. Complete non-overlapping blocks of ceil(g) points are averaged. "
        "Block OLS, residual normality, and Spearman time correlation are then "
        "calculated on these approximately decorrelated block means. If fewer "
        "than three complete blocks remain, block inference is not calculated.\n"
    )
    report.write(
        "4. Benjamini-Hochberg FDR-adjusted p-values account for simultaneous "
        "testing across all extracted NAMD parameters.\n"
    )
    report.write(
        "5. Shapiro-Wilk p > 0.05 means normality was not rejected; it does not "
        "prove normality. Durbin-Watson near 2 and Ljung-Box p > 0.05 support "
        "weak residual serial dependence.\n"
    )
    report.write(
        "6. Frames are not independent experimental replicates. Independent "
        "simulation replicas remain necessary for population-level biological "
        "inference or direct comparisons between systems.\n\n"
    )

    for _, row in temporal_statistics.iterrows():
        parameter = row.get("Parameter", "Unknown")
        report.write(f"PARAMETER: {parameter}\n")
        report.write("-" * 78 + "\n")

        if row.get("Analysis_status") != "OK":
            report.write(f"Status: {row.get('Analysis_status')}\n\n")
            continue

        report.write(
            "Descriptive: "
            f"N={int(row['N_raw'])}; "
            f"mean={format_number(row['Mean'])}; "
            f"SD={format_number(row['SD'])}; "
            f"median={format_number(row['Median'])}; "
            f"IQR={format_number(row['IQR'])}.\n"
        )
        report.write(
            "Temporal dependence: "
            f"ACF(1)={format_number(row['Metric_ACF_lag1'])}; "
            f"g={format_number(row['Statistical_inefficiency_g'])}; "
            f"effective N={format_number(row['Effective_sample_size'])}; "
            f"block size={int(row['Automatic_block_size_points'])} points; "
            f"complete blocks={int(row['N_complete_blocks'])}.\n"
        )
        report.write(
            "Raw OLS + HAC: "
            f"slope={format_number(row['Raw_OLS_slope_per_time_unit'])} "
            f"per {statistical_time_unit}; "
            f"HAC SE={format_number(row['HAC_slope_SE'])}; "
            f"95% CI=[{format_number(row['HAC_slope_CI95_low'])}, "
            f"{format_number(row['HAC_slope_CI95_high'])}]; "
            f"p={format_number(row['HAC_slope_p'])}; "
            f"FDR p={format_number(row['HAC_slope_p_FDR_BH'])}; "
            f"R2={format_number(row['Raw_OLS_R2'])}; "
            f"adjusted R2={format_number(row['Raw_OLS_adjusted_R2'])}; "
            f"RMSE={format_number(row['Raw_OLS_RMSE'])}; "
            f"AIC={format_number(row['Raw_OLS_AIC'])}; "
            f"BIC={format_number(row['Raw_OLS_BIC'])}.\n"
        )
        report.write(
            "Raw residual diagnostics: "
            f"ACF(1)={format_number(row['Raw_residual_ACF_lag1'])}; "
            f"Durbin-Watson={format_number(row['Raw_residual_Durbin_Watson'])}; "
            f"Ljung-Box p={format_number(row['Raw_residual_Ljung_Box_p'])} "
            f"at lag {int(row['Raw_residual_Ljung_Box_lag'])}.\n"
        )
        report.write(
            "Block OLS: "
            f"slope={format_number(row['Block_OLS_slope_per_time_unit'])} "
            f"per {statistical_time_unit}; "
            f"SE={format_number(row['Block_OLS_slope_SE'])}; "
            f"95% CI=[{format_number(row['Block_OLS_slope_CI95_low'])}, "
            f"{format_number(row['Block_OLS_slope_CI95_high'])}]; "
            f"p={format_number(row['Block_OLS_slope_p'])}; "
            f"FDR p={format_number(row['Block_OLS_slope_p_FDR_BH'])}; "
            f"R2={format_number(row['Block_OLS_R2'])}; "
            f"adjusted R2={format_number(row['Block_OLS_adjusted_R2'])}; "
            f"RMSE={format_number(row['Block_OLS_RMSE'])}.\n"
        )
        report.write(
            "Block residual normality: "
            f"Shapiro-Wilk W={format_number(row['Block_residual_Shapiro_W'])}; "
            f"p={format_number(row['Block_residual_Shapiro_p'])}; "
            f"N tested={int(row['Block_residual_Shapiro_N'])}; "
            f"Jarque-Bera={format_number(row['Block_residual_Jarque_Bera'])}; "
            f"p={format_number(row['Block_residual_Jarque_Bera_p'])}.\n"
        )
        if int(row["Block_residual_Shapiro_N"]) < 8:
            report.write(
                "Normality caution: fewer than 8 block residuals provide very "
                "limited power for distributional assessment.\n"
            )
        report.write(
            "Block Spearman versus time: "
            f"rho={format_number(row['Block_Spearman_rho_time'])}; "
            f"p={format_number(row['Block_Spearman_p'])}; "
            f"FDR p={format_number(row['Block_Spearman_p_FDR_BH'])}; "
            f"p-value method={row['Block_Spearman_p_method']}.\n\n"
        )


print(
    f"\n[OK] Temporal statistics CSV saved:\n"
    f"{output_temporal_statistics_csv}"
)

print(
    f"\n[OK] Temporal statistics text report saved:\n"
    f"{output_temporal_statistics_txt}"
)


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
MMV675968_energy_temporal_statistics.csv
MMV675968_energy_temporal_statistics.txt

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
