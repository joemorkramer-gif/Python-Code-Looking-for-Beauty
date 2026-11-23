import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
import os
import glob


# look at the CSV-Files
csv_files = glob.glob("behavioral summary/*.csv")
print(f"Found {len(csv_files)} CSV files")

### Raw Data ###

# list to store all dataframes
all_data = []

# Loop through each CSV file
for file in csv_files:
    df = pd.read_csv(file)
    all_data.append(df)

print(all_data)

## Averaging ##
# Aggregate by artwork_id across all participants
artwork_data = {}

for df in all_data:
    for idx, row in df.iterrows():
        artwork_id = row["artwork_id"]

        if artwork_id not in artwork_data:
            artwork_data[artwork_id] = {
                "rating": [],
                "n_fixations": [],
                "n_blinks": [],
                "time_to_first_fixation": [],
            }

        artwork_data[artwork_id]["rating"].append(row["rating"])
        artwork_data[artwork_id]["n_fixations"].append(row["n_fixations"])
        artwork_data[artwork_id]["n_blinks"].append(row["n_blinks"])
        artwork_data[artwork_id]["time_to_first_fixation"].append(
            row["time_to_first_fixation"]
        )


averaged_list = []
for artwork_id, values in artwork_data.items():
    averaged_list.append(
        {
            "artwork_id": artwork_id,
            "avg_rating": np.mean(values["rating"]),
            "avg_n_fixations": np.mean(values["n_fixations"]),
            "avg_n_blinks": np.mean(values["n_blinks"]),
            "avg_time_to_first_fixation": np.mean(values["time_to_first_fixation"]),
        }
    )

# Create averaged dataframe
averaged_df = pd.DataFrame(averaged_list)
print(averaged_df)

## Correlations ##

# Spearman-correlations
corr_fix, p_fix = spearmanr(averaged_df["avg_rating"], averaged_df["avg_n_fixations"])
corr_blink, p_blink = spearmanr(averaged_df["avg_rating"], averaged_df["avg_n_blinks"])
corr_ttff, p_ttff = spearmanr(
    averaged_df["avg_rating"], averaged_df["avg_time_to_first_fixation"]
)

print(f"Fixations:      r={corr_fix:.4f}, p={p_fix:.4f}")
# Fixations:      r=0.0127, p=0.3447
print(f"Blinks:         r={corr_blink:.4f}, p={p_blink:.4f}")
# Blinks:         r=-0.0161, p=0.2304
print(f"Time to first:  r={corr_ttff:.4f}, p={p_ttff:.4f}")
# Time to first:  r=-0.0095, p=0.4774

# Pearson-correlations
pcorr_fix, pp_fix = pearsonr(averaged_df["avg_rating"], averaged_df["avg_n_fixations"])
pcorr_blink, pp_blink = pearsonr(averaged_df["avg_rating"], averaged_df["avg_n_blinks"])
pcorr_ttff, pp_ttff = pearsonr(
    averaged_df["avg_rating"], averaged_df["avg_time_to_first_fixation"]
)

print(f"Fixations:      r={pcorr_fix:.4f}, p={pp_fix:.4f}")
# Fixations:      r=0.0064, p=0.6350
print(f"Blinks:         r={pcorr_blink:.4f}, p={pp_blink:.4f}")
# Blinks:         r=-0.0227, p=0.0900
print(f"Time to first:  r={pcorr_ttff:.4f}, p={pp_ttff:.4f}")
# Time to first:  r=-0.0098, p=0.4628


# Scatter plots

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Correlations with Per-Artwork Average Ratings", fontsize=14)

# Fixations
axes[0].scatter(
    averaged_df["avg_n_fixations"],
    averaged_df["avg_rating"],
    alpha=0.6,
    edgecolors="black",
    linewidth=0.5,
)
axes[0].set_xlabel("Average Number of Fixations")
axes[0].set_ylabel("Average Rating")
axes[0].set_title(f"Hypothesis 1a: Fixation Count")

# Blinks
axes[1].scatter(
    averaged_df["avg_n_blinks"],
    averaged_df["avg_rating"],
    alpha=0.6,
    edgecolors="black",
    linewidth=0.5,
)
axes[1].set_xlabel("Average Number of Blinks")
axes[1].set_ylabel("Average Rating")
axes[1].set_title(f"Hypothesis 1b: Blink Count")

# Time to first fixation
axes[2].scatter(
    averaged_df["avg_time_to_first_fixation"],
    averaged_df["avg_rating"],
    alpha=0.6,
    edgecolors="black",
    linewidth=0.5,
)
axes[2].set_xlabel("Average Time to First Fixation (s)")
axes[2].set_ylabel("Average Rating")
axes[2].set_title(f"Hypothesis 1c: Time to First Fixation")

plt.tight_layout()
# fig.savefig("raw_correlations.png", dpi=300, bbox_inches="tight")
plt.show()


### Z-scored data ###

## Z-scoring ##
from scipy.stats import spearmanr, zscore

# Create a new list to store z-scored data for each participant
zscored_all_data = []

for df in all_data:
    # Create a copy to avoid modifying original
    df_z = df.copy()

    # Z-score each measure
    df_z["rating_z"] = zscore(df["rating"])
    df_z["n_fixations_z"] = zscore(df["n_fixations"])
    df_z["n_blinks_z"] = zscore(df["n_blinks"])
    df_z["time_to_first_fixation_z"] = zscore(df["time_to_first_fixation"])

    # Add to list
    zscored_all_data.append(df_z)

print(df_z)
print(zscored_all_data)

## Averaging ##

artwork_data_z = {}

for df_z in zscored_all_data:
    for idx, row in df_z.iterrows():
        artwork_id = row["artwork_id"]

        if artwork_id not in artwork_data_z:
            artwork_data_z[artwork_id] = {
                "rating_z": [],
                "n_fixations_z": [],
                "n_blinks_z": [],
                "time_to_first_fixation_z": [],
            }

        artwork_data_z[artwork_id]["rating_z"].append(row["rating_z"])
        artwork_data_z[artwork_id]["n_fixations_z"].append(row["n_fixations_z"])
        artwork_data_z[artwork_id]["n_blinks_z"].append(row["n_blinks_z"])
        artwork_data_z[artwork_id]["time_to_first_fixation_z"].append(
            row["time_to_first_fixation_z"]
        )


averaged_list_z = []
for artwork_id, values in artwork_data_z.items():
    averaged_list_z.append(
        {
            "artwork_id": artwork_id,
            "avg_rating_z": np.mean(values["rating_z"]),
            "avg_n_fixations_z": np.mean(values["n_fixations_z"]),
            "avg_n_blinks_z": np.mean(values["n_blinks_z"]),
            "avg_time_to_first_fixation_z": np.mean(values["time_to_first_fixation_z"]),
        }
    )

# Create averaged dataframe
averaged_df_z = pd.DataFrame(averaged_list_z)


## Correlations ##

# Spearman-correlations
corr_fix_z, p_fix_z = spearmanr(
    averaged_df_z["avg_rating_z"], averaged_df_z["avg_n_fixations_z"]
)
corr_blink_z, p_blink_z = spearmanr(
    averaged_df_z["avg_rating_z"], averaged_df_z["avg_n_blinks_z"]
)
corr_ttff_z, p_ttff_z = spearmanr(
    averaged_df_z["avg_rating_z"], averaged_df_z["avg_time_to_first_fixation_z"]
)

print(f"Fixations:      r={corr_fix_z:.4f}, p={p_fix_z:.4f}")
# Fixations:      r=0.1557, p=0.0369
print(f"Blinks:         r={corr_blink_z:.4f}, p={p_blink_z:.4f}")
# Blinks:         r=0.1012, p=0.1766
print(f"Time to first:  r={corr_ttff_z:.4f}, p={p_ttff_z:.4f}")
# Time to first:  r=0.0370, p=0.6218

# Pearson-correlations
pcorr_fix_z, pp_fix_z = pearsonr(
    averaged_df_z["avg_rating_z"], averaged_df_z["avg_n_fixations_z"]
)
pcorr_blink_z, pp_blink_z = pearsonr(
    averaged_df_z["avg_rating_z"], averaged_df_z["avg_n_blinks_z"]
)
pcorr_ttff_z, pp_ttff_z = pearsonr(
    averaged_df_z["avg_rating_z"], averaged_df_z["avg_time_to_first_fixation_z"]
)

print(f"Fixations:      r={pcorr_fix_z:.4f}, p={pp_fix_z:.4f}")
# Fixations:      r=0.1663, p=0.0256
print(f"Blinks:         r={pcorr_blink_z:.4f}, p={pp_blink_z:.4f}")
# Blinks:         r=0.0299, p=0.6906
print(f"Time to first:  r={pcorr_ttff_z:.4f}, p={pp_ttff_z:.4f}")
# Time to first:  r=-0.0255, p=0.7344

# Scatter plots

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Z-scored Correlations with Per-Artwork Average Ratings", fontsize=14)

# Fixations
axes[0].scatter(
    averaged_df_z["avg_n_fixations_z"],
    averaged_df_z["avg_rating_z"],
    alpha=0.6,
    edgecolors="black",
    linewidth=0.5,
)
axes[0].set_xlabel("Average Number of Fixations")
axes[0].set_ylabel("Average Rating")
axes[0].set_title(f"Fixations: Spearman ρ={corr_fix_z:.3f}, p={p_fix_z:.3f}")

# Blinks
axes[1].scatter(
    averaged_df_z["avg_n_blinks_z"],
    averaged_df_z["avg_rating_z"],
    alpha=0.6,
    edgecolors="black",
    linewidth=0.5,
)
axes[1].set_xlabel("Average Number of Blinks")
axes[1].set_ylabel("Average Rating")
axes[1].set_title(f"Blinks: Spearman ρ={corr_blink_z:.3f}, p={p_blink_z:.3f}")

# Time to first fixation
axes[2].scatter(
    averaged_df_z["avg_time_to_first_fixation_z"],
    averaged_df_z["avg_rating_z"],
    alpha=0.6,
    edgecolors="black",
    linewidth=0.5,
)
axes[2].set_xlabel("Average Time to First Fixation (s)")
axes[2].set_ylabel("Average Rating")
axes[2].set_title(f"Time to First: Spearman ρ={corr_ttff_z:.3f}, p={p_ttff_z:.3f}")

plt.tight_layout()
# fig.savefig("zscored_correlations.png", dpi=300, bbox_inches="tight")
plt.show()


### Linear Regression Models ###

# raw averaged data
import statsmodels.formula.api as smf
import pandas as pd

model = smf.ols(
    "avg_rating ~ avg_n_fixations + avg_n_blinks + avg_time_to_first_fixation",
    data=averaged_df,
).fit()


# View the results
print(model.summary())

# z-scored data

model_z = smf.ols(
    "avg_rating_z ~ avg_n_fixations_z + avg_n_blinks_z + avg_time_to_first_fixation_z",
    data=averaged_df_z,
).fit()


# View the results
print(model_z.summary())


### Presentation-ready scatter plot ###

# Set font sizes and figure parameters for presentation
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

# Create figure with smaller size for presentation
fig, axes = plt.subplots(1, 3, figsize=(12, 4))  # Smaller figure size
fig.suptitle("Z-scored Correlations with Per-Artwork Average Ratings", fontsize=18, fontweight='bold')

# Fixations
axes[0].scatter(
    averaged_df_z["avg_n_fixations_z"],
    averaged_df_z["avg_rating_z"],
    alpha=0.6,
    edgecolors="black",
    linewidth=0.5,
    s=80,  # Slightly larger points
)
axes[0].set_xlabel("Z-scored Number of Fixations", fontsize=14)
axes[0].set_ylabel("Z-scored Rating", fontsize=14)
axes[0].set_title(f"Fixations: ρ={corr_fix_z:.3f}, p={p_fix_z:.3f}", fontsize=14)
axes[0].tick_params(axis='both', which='major', labelsize=12)

# Blinks
axes[1].scatter(
    averaged_df_z["avg_n_blinks_z"],
    averaged_df_z["avg_rating_z"],
    alpha=0.6,
    edgecolors="black",
    linewidth=0.5,
    s=80,
)
axes[1].set_xlabel("Z-scored Number of Blinks", fontsize=14)
axes[1].set_ylabel("Z-scored Rating", fontsize=14)
axes[1].set_title(f"Blinks: ρ={corr_blink_z:.3f}, p={p_blink_z:.3f}", fontsize=14)
axes[1].tick_params(axis='both', which='major', labelsize=12)

# Time to first fixation
axes[2].scatter(
    averaged_df_z["avg_time_to_first_fixation_z"],
    averaged_df_z["avg_rating_z"],
    alpha=0.6,
    edgecolors="black",
    linewidth=0.5,
    s=80,
)
axes[2].set_xlabel("Z-scored Time to First Fixation", fontsize=14)
axes[2].set_ylabel("Z-scored Rating", fontsize=14)
axes[2].set_title(f"Time to First: ρ={corr_ttff_z:.3f}, p={p_ttff_z:.3f}", fontsize=14)
axes[2].tick_params(axis='both', which='major', labelsize=12)

# Adjust layout and save with high DPI
plt.tight_layout()
# fig.savefig("zscored_correlations.png", dpi=600, bbox_inches="tight")  
plt.show()


# OLS
model = smf.ols("avg_rating ~ avg_n_fixations + avg_n_blinks + avg_time_to_first_fixation",
                data=averaged_df).fit()
print(model.df_model, model.df_resid)