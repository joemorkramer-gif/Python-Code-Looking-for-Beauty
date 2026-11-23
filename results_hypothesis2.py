import pandas as pd
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, zscore, ttest_1samp
from scipy.spatial.distance import pdist, squareform
import glob
import os
import multimatch_gaze as m
from scipy.stats import spearmanr, zscore, ttest_1samp, shapiro
import seaborn as sns

##### Hypothesis 2a-2c: Fixation count, blink count, time to first fixation #####

# Load CSV files
os.getcwd()
os.chdir(
    "/Users/joemorkramer/Documents/Uni/SoSe 26/Seafile/Beauty and eyes/behavioral summary"
)
csv_files = glob.glob("*.csv")
print(f"Found {len(csv_files)} CSV files")

# Load all data and z-score per participant
all_data_zscored = []

for file in csv_files:
    df = pd.read_csv(file)

    # Create copy and z-score within each participant
    df_z = df.copy()
    df_z["rating_z"] = zscore(df["rating"])
    df_z["n_fixations_z"] = zscore(df["n_fixations"])
    df_z["n_blinks_z"] = zscore(df["n_blinks"])
    df_z["time_to_first_fixation_z"] = zscore(df["time_to_first_fixation"])

    all_data_zscored.append(df_z)

print(f"Loaded and z-scored data for {len(all_data_zscored)} participants")

# participant IDs
participant_ids = [f"P{i + 1}" for i in range(len(all_data_zscored))]
n_participants = len(all_data_zscored)

# artwork IDs
artwork_ids = sorted(all_data_zscored[0]["artwork_id"].unique())
n_artworks = len(artwork_ids)

print(f"Number of participants: {n_participants}")
print(f"Number of artworks: {n_artworks}")


### Organize z-scored data ###

artwork_data = {}
for artwork_id in artwork_ids:
    artwork_data[artwork_id] = {
        "participants": [],
        "ratings_z": [],
        "fixations_z": [],
        "blinks_z": [],
        "ttff_z": [],
    }

# Fill structure with z-scored values
for p_idx, df in enumerate(all_data_zscored):
    participant_id = participant_ids[p_idx]

    for idx, row in df.iterrows():
        artwork_id = row["artwork_id"]

        artwork_data[artwork_id]["participants"].append(participant_id)
        artwork_data[artwork_id]["ratings_z"].append(row["rating_z"])
        artwork_data[artwork_id]["fixations_z"].append(row["n_fixations_z"])
        artwork_data[artwork_id]["blinks_z"].append(row["n_blinks_z"])
        artwork_data[artwork_id]["ttff_z"].append(row["time_to_first_fixation_z"])


# Store correlation results for each artwork
correlation_results = []

for artwork_id in artwork_ids:
    # Get z-scored values as numpy arrays
    ratings_z = np.array(artwork_data[artwork_id]["ratings_z"])[:, np.newaxis]
    fixations_z = np.array(artwork_data[artwork_id]["fixations_z"])[:, np.newaxis]
    blinks_z = np.array(artwork_data[artwork_id]["blinks_z"])[:, np.newaxis]
    ttff_z = np.array(artwork_data[artwork_id]["ttff_z"])[:, np.newaxis]

    # Compute condensed RDMs using pdist
    rdm_rating = pdist(ratings_z, metric=lambda u, v: u[0] - v[0])
    rdm_fixation = pdist(fixations_z, metric=lambda u, v: u[0] - v[0])
    rdm_blink = pdist(blinks_z, metric=lambda u, v: u[0] - v[0])
    rdm_ttff = pdist(ttff_z, metric=lambda u, v: u[0] - v[0])

    # Correlate rating RDM with each eye movement RDM
    corr_fix, p_fix = spearmanr(rdm_rating, rdm_fixation)
    corr_blink, p_blink = spearmanr(rdm_rating, rdm_blink)
    corr_ttff, p_ttff = spearmanr(rdm_rating, rdm_ttff)

    correlation_results.append(
        {
            "artwork_id": artwork_id,
            "corr_fixation": corr_fix,
            "p_fixation": p_fix,
            "corr_blink": corr_blink,
            "p_blink": p_blink,
            "corr_ttff": corr_ttff,
            "p_ttff": p_ttff,
        }
    )

# Convert to DataFrame
corr_df = pd.DataFrame(correlation_results)

print("\nCorrelation Results for Each Artwork (Z-scored)")
print(corr_df.head(10))


### Summary statistics across artworks

print("\nSummary Statistics Across All Artworks (Z-scored)")
for metric in ["fixation", "blink", "ttff"]:
    corr_col = f"corr_{metric}"
    p_col = f"p_{metric}"

    mean_corr = corr_df[corr_col].mean()
    std_corr = corr_df[corr_col].std()
    n_significant = (corr_df[p_col] < 0.05).sum()

    print(f"\n{metric.upper()}:")
    print(f"  Mean correlation: {mean_corr:.4f} (±{std_corr:.4f})")
    print(f"  Range: [{corr_df[corr_col].min():.4f}, {corr_df[corr_col].max():.4f}]")
    print(
        f"  Number of significant correlations (p < 0.05): {n_significant}/{len(corr_df)}"
    )



### Sample t-test

## Before t-test, check for normal distribution using Shapiro-Wilk test

from scipy.stats import shapiro

print("\nShapiro-Wilk Test for Normality (Z-scored)")
for metric in ["fixation", "blink", "ttff"]:
    corr_col = f"corr_{metric}"
    
    # Run Shapiro-Wilk on the correlations
    statistic, p_val = shapiro(corr_df[corr_col])
    
    print(f"{metric.upper()}: W={statistic:.4f}, p={p_val:.4f}")

# If the Shapiro-Wilk test shows no significance for all three sub-hypotheses, 
# we can proceed to the t-test.

## t-test
print("\nOne-sample t-test against 0 (Z-scored)")
for metric in ["fixation", "blink", "ttff"]:
    corr_col = f"corr_{metric}"
    t_stat, p_val = ttest_1samp(corr_df[corr_col], 0)
    print(f"{metric.upper()}: t={t_stat:.4f}, p={p_val:.4f}")


### Swarm plot 

swarm_data = []

for _, row in corr_df.iterrows():
    swarm_data.append({
        'metric': 'Fixation',
        'correlation': row['corr_fixation']
    })
    swarm_data.append({
        'metric': 'Blink',
        'correlation': row['corr_blink']
    })
    swarm_data.append({
        'metric': 'TTFF',
        'correlation': row['corr_ttff']
    })

swarm_df = pd.DataFrame(swarm_data)

# Create swarm plot
plt.figure(figsize=(10, 6))

sns.swarmplot(
    data=swarm_df,
    x='metric',
    y='correlation',
    palette=['blue', 'green', 'red'],
    size=8,
    alpha=0.7,
    edgecolor='black',
    linewidth=0.5
)

plt.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)

plt.xlabel('Eye Movement Metric', fontsize=12)
plt.ylabel('Spearman ρ (RDM Correlation)', fontsize=12)
plt.title('Distribution of RDM Correlations Across Artworks', fontsize=14)

plt.ylim(-1, 1)
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# Print summary statistics for reference
print("\nSummary statistics for swarm plot data:")
print(swarm_df.groupby('metric')['correlation'].agg(['mean', 'std', 'min', 'max', 'count']))

##### Hypothesis 2d: Scanpaths #####


### configuration ###
BEHAVIORAL_DATA_PATH = r"/Users/joemorkramer/Documents/Uni/SoSe 26/Seafile/Beauty and eyes/behavioral summary"
SCANPATH_DATA_PATH = r"/Users/joemorkramer/Documents/Uni/SoSe 26/Seafile/Beauty and eyes/scanpath"
SCREEN_WIDTH = 1200  
SCREEN_HEIGHT = 800 

# Define MultiMatch dimension names
MM_DIMENSIONS = ['vector', 'direction', 'length', 'position', 'duration']



### step 1: participant list from scanpath folders ###
# so that we have a list off all participant ids to compare
subject_folders = glob.glob(os.path.join(SCANPATH_DATA_PATH, "sub-EA*"))
print(f"Found {len(subject_folders)} subject folders with scanpath data")

scanpath_participant_ids = [os.path.basename(f) for f in subject_folders]
scanpath_participant_ids.sort()
print(f"Scanpath participant IDs: {scanpath_participant_ids[:5]}...")


#### step 2: load behavioral data ###
## Change working directory and find csv files 
os.chdir(BEHAVIORAL_DATA_PATH)
behavioral_files = glob.glob("*.csv")
print(f"\nFound {len(behavioral_files)} behavioral CSV files")

## Check first file for column names
# optional, just to help debug in case there's errors 
sample_df = pd.read_csv(behavioral_files[0])
print(f"Columns in behavioral files: {sample_df.columns.tolist()}")

## Load behavioral data
# loop through the behavioral csv-files and create z-scored copies
all_behavioral_data = []
behavioral_participant_ids = []

for file in behavioral_files:
    df = pd.read_csv(file)
    
    # Extract participant ID
    participant_id = os.path.splitext(file)[0]
    participant_id = participant_id.replace("_summary", "")
    behavioral_participant_ids.append(participant_id)
    
    # Z-score ratings (assuming column is 'rating')
    df_z = df.copy()
    df_z["rating_z"] = zscore(df["rating"])
    all_behavioral_data.append(df_z)

print(f"Loaded behavioral data for {len(all_behavioral_data)} participants")


### step 3: match participants ###
# make sure the participants from behavioral summary folder and scanpath folder match 
# and create numeric index

valid_participants = list(set(behavioral_participant_ids) & set(scanpath_participant_ids))
valid_participants.sort()
print(f"\nParticipants with both behavioral and scanpath data: {len(valid_participants)}")

if len(valid_participants) == 0:
    print("\nERROR: No matching participants found!")
    print("Scanpath folders:", scanpath_participant_ids[:5])
    print("Behavioral IDs:", behavioral_participant_ids[:5])
    exit()

participant_to_idx = {pid: idx for idx, pid in enumerate(valid_participants)}

# Get artwork IDs
first_participant = valid_participants[0]
first_idx = behavioral_participant_ids.index(first_participant)
artwork_ids = sorted(all_behavioral_data[first_idx]["artwork_id"].unique())
print(f"Found {len(artwork_ids)} unique artworks")


### step 4: load scanpath data ###
# turn in format that MultiMatch expects 
def load_scanpath_file(file_path):
    """Load a scanpath file and return structured array for MultiMatch"""
    try:
        if file_path.endswith('.tsv'):
            df = pd.read_csv(file_path, sep='\t')
        else:
            df = pd.read_csv(file_path)
        
        # column names
        x_col = 'x'
        y_col = 'y'
        dur_col = 'duration [ms]'
        
        if x_col not in df.columns or y_col not in df.columns or dur_col not in df.columns:
            print(f"  Missing columns in {file_path}")
            return None
        
        if len(df) < 2:
            return None
        
        # Convert duration from ms to seconds because MultiMatch expects seconds 
        df[dur_col] = df[dur_col] / 1000
        
        vectors = np.rec.fromarrays(
            (df[x_col].values.astype(float),
             df[y_col].values.astype(float),
             df[dur_col].values.astype(float)),
            dtype=[('start_x', 'f8'), ('start_y', 'f8'), ('duration', 'f8')]
        )
        return vectors
    # structured NumPy array with dtype to define column names and data types for the format docomparison() needs
        
    except Exception as e:
        print(f"  Error loading {file_path}: {e}")
        return None

# Dictionary: scanpath_vectors[artwork_id][participant_idx] = fixation_array
scanpath_vectors = {}

for participant_id in valid_participants:
    p_idx = participant_to_idx[participant_id]
    participant_folder = os.path.join(SCANPATH_DATA_PATH, participant_id)
    
    if not os.path.exists(participant_folder):
        print(f"Warning: Folder not found for {participant_id}")
        continue
    
    scanpath_files = glob.glob(os.path.join(participant_folder, "*.csv"))
    scanpath_files.extend(glob.glob(os.path.join(participant_folder, "*.tsv")))
    
    for file_path in scanpath_files:
        file_name = os.path.basename(file_path)
        artwork_id = os.path.splitext(file_name)[0]
        
        # Remove prefixes
        for prefix in ["artwork_", "img_", "fix_", "image_", "stim_", "scanpath_"]:
            if artwork_id.startswith(prefix):
                artwork_id = artwork_id[len(prefix):]
        
        if artwork_id not in artwork_ids:
            continue
        
        vectors = load_scanpath_file(file_path)
        if vectors is None:
            continue
        
        if artwork_id not in scanpath_vectors:
            scanpath_vectors[artwork_id] = {}
        scanpath_vectors[artwork_id][p_idx] = vectors

print(f"\nLoaded scanpath data for {len(scanpath_vectors)} artworks")


### step 5: organize rating data ### 
# loop through participants and organizes standardized ratings for any artwork and participant

rating_data = {artwork_id: {} for artwork_id in artwork_ids}

for participant_id in valid_participants:
    p_idx = participant_to_idx[participant_id]
    behavioral_idx = behavioral_participant_ids.index(participant_id)
    df = all_behavioral_data[behavioral_idx]
    
    for _, row in df.iterrows():
        artwork_id = row["artwork_id"]
        rating_data[artwork_id][p_idx] = row["rating_z"]


### step 6: compute scanpath dissimilarity matrix ###

print("\n" + "="*60)
print("SCANPATH DISSIMILARITY ANALYSIS - ALL 5 DIMENSIONS")
print("="*60)

# Store results for each dimension separately
results_by_dimension = {dim: [] for dim in MM_DIMENSIONS}

for artwork_id in artwork_ids:
    if artwork_id not in scanpath_vectors:
        continue
    
    available_participants = sorted(scanpath_vectors[artwork_id].keys())
    available_participants = [p for p in available_participants if p in rating_data[artwork_id]]
    
    n_available = len(available_participants)
    if n_available < 2:
        continue
    
    print(f"\nProcessing artwork {artwork_id}: {n_available} participants")
    
    # --- RATING DISSIMILARITY MATRIX ---
    ratings = np.array([rating_data[artwork_id][p] for p in available_participants])
    ratings = ratings[:, np.newaxis]
    rdm_rating_dissimilarity = pdist(ratings, metric=lambda u, v: u[0] - v[0])
    
    # --- SCANPATH DISSIMILARITY MATRIX FOR EACH DIMENSION ---
    n_pairs = n_available * (n_available - 1) // 2
    
    # Initialize arrays for each dimension
    dim_dissimilarities = {dim: np.zeros(n_pairs) for dim in MM_DIMENSIONS}
    
    pair_idx = 0
    for i_idx, i in enumerate(available_participants):
        for j in available_participants[i_idx + 1:]:
            vec_i = scanpath_vectors[artwork_id][i]
            vec_j = scanpath_vectors[artwork_id][j]
            
            try:
                # Get all 5 MultiMatch scores
                scores = m.docomparison(
                    vec_i, vec_j,
                    screensize=[SCREEN_WIDTH, SCREEN_HEIGHT],
                    grouping=False
                )
                # scores = [vector, direction, length, position, duration]
                
                # Store each dimension's dissimilarity (1 - similarity)
                for dim_idx, dim_name in enumerate(MM_DIMENSIONS):
                    dim_dissimilarities[dim_name][pair_idx] = 1 - scores[dim_idx]
                
            except Exception as e:
                print(f"  Error comparing P{i} vs P{j}: {e}")
                # Mark all dimensions as NaN for this pair
                for dim_name in MM_DIMENSIONS:
                    dim_dissimilarities[dim_name][pair_idx] = np.nan
            
            pair_idx += 1
    
    # For each dimension, correlate with rating RDM
    for dim_name in MM_DIMENSIONS:
        dim_dissimilarity = dim_dissimilarities[dim_name]
        
        # Remove NaNs
        valid = ~np.isnan(dim_dissimilarity) & ~np.isnan(rdm_rating_dissimilarity)
        n_valid = valid.sum()
        
        if n_valid < 2:
            print(f"  ⚠️ Not enough valid pairs for {dim_name} (only {n_valid} valid)")
            results_by_dimension[dim_name].append({
                "artwork_id": artwork_id,
                "correlation": np.nan,
                "p_value": np.nan,
                "n_participants": n_available,
                "n_pairs_total": n_pairs,
                "n_pairs_valid": n_valid
            })
            continue
        
        # Calculate correlation
        corr, p_val = spearmanr(
            rdm_rating_dissimilarity[valid], 
            dim_dissimilarity[valid]
        )
        
        # Check if correlation is NaN (can happen if all values are identical)
        if np.isnan(corr):
            print(f"  ⚠️ Correlation is NaN for {dim_name}")
        
        results_by_dimension[dim_name].append({
            "artwork_id": artwork_id,
            "correlation": corr,
            "p_value": p_val,
            "n_participants": n_available,
            "n_pairs_total": n_pairs,
            "n_pairs_valid": n_valid
        })
        
        print(f"  {dim_name.capitalize()}: ρ = {corr:.4f}, p = {p_val:.4f} (using {n_valid}/{n_pairs} pairs)")


# step 6b: compute scanpath dissimilarity matrix averaged across the 5 dimensions ###

print("\n" + "="*60)
print("SCANPATH DISSIMILARITY ANALYSIS - AVERAGED ACROSS 5 DIMENSIONS")
print("="*60)

results_averaged = []

for artwork_id in artwork_ids:
    if artwork_id not in scanpath_vectors:
        continue
    
    available_participants = sorted(scanpath_vectors[artwork_id].keys())
    available_participants = [p for p in available_participants if p in rating_data[artwork_id]]
    
    n_available = len(available_participants)
    if n_available < 2:
        continue
    
    print(f"\nProcessing artwork {artwork_id}: {n_available} participants")
    
    # --- RATING DISSIMILARITY MATRIX ---
    ratings = np.array([rating_data[artwork_id][p] for p in available_participants])
    ratings = ratings[:, np.newaxis]
    rdm_rating_dissimilarity = pdist(ratings, metric=lambda u, v: u[0] - v[0])
    
    # --- SCANPATH DISSIMILARITY MATRIX (AVERAGED) ---
    n_pairs = n_available * (n_available - 1) // 2
    avg_dissimilarities = np.zeros(n_pairs)
    
    pair_idx = 0
    for i_idx, i in enumerate(available_participants):
        for j in available_participants[i_idx + 1:]:
            vec_i = scanpath_vectors[artwork_id][i]
            vec_j = scanpath_vectors[artwork_id][j]
            
            try:
                # Get all 5 MultiMatch scores
                scores = m.docomparison(
                    vec_i, vec_j,
                    screensize=[SCREEN_WIDTH, SCREEN_HEIGHT],
                    grouping=False
                )
                # Average the 5 similarity scores, then convert to dissimilarity
                avg_similarity = np.mean(scores)
                avg_dissimilarities[pair_idx] = 1 - avg_similarity
                
            except Exception as e:
                print(f"  Error comparing P{i} vs P{j}: {e}")
                avg_dissimilarities[pair_idx] = np.nan
            
            pair_idx += 1
    
    # Correlate averaged scanpath RDM with rating RDM
    valid = ~np.isnan(avg_dissimilarities) & ~np.isnan(rdm_rating_dissimilarity)
    n_valid = valid.sum()
    
    if n_valid < 2:
        print(f"  ⚠️ Not enough valid pairs (only {n_valid} valid)")
        results_averaged.append({
            "artwork_id": artwork_id,
            "correlation": np.nan,
            "p_value": np.nan,
            "n_participants": n_available,
            "n_pairs_total": n_pairs,
            "n_pairs_valid": n_valid
        })
        continue
    
    corr, p_val = spearmanr(
        rdm_rating_dissimilarity[valid],
        avg_dissimilarities[valid]
    )
    
    if np.isnan(corr):
        print(f"  ⚠️ Correlation is NaN")
    
    results_averaged.append({
        "artwork_id": artwork_id,
        "correlation": corr,
        "p_value": p_val,
        "n_participants": n_available,
        "n_pairs_total": n_pairs,
        "n_pairs_valid": n_valid
    })
    
    print(f"  Averaged: ρ = {corr:.4f}, p = {p_val:.4f} (using {n_valid}/{n_pairs} pairs)")



# step 7: results and statistics 
print("\n" + "="*60)
print("RESULTS FOR EACH DIMENSION")
print("="*60)

# Store all DataFrames for later use
dfs_by_dimension = {}

for dim_name in MM_DIMENSIONS:
    df = pd.DataFrame(results_by_dimension[dim_name])
    dfs_by_dimension[dim_name] = df
    
    # Remove NaN correlations for statistics
    clean_df = df.dropna(subset=['correlation'])
    n_removed = len(df) - len(clean_df)
    
    print(f"\n{'='*40}")
    print(f"DIMENSION: {dim_name.upper()}")
    print(f"{'='*40}")
    
    if len(clean_df) == 0:
        print(f"  No valid correlations (all NaN)")
        print(f"  Total artworks: {len(df)}")
        print(f"  With NaN: {n_removed}")
        continue
    
    print(f"  Total artworks: {len(df)}")
    if n_removed > 0:
        print(f"  Removed {n_removed} artworks with NaN correlations")
    
    print(f"\n  Summary Statistics:")
    print(f"    Mean correlation: {clean_df['correlation'].mean():.4f}")
    print(f"    Std deviation: {clean_df['correlation'].std():.4f}")
    print(f"    Range: [{clean_df['correlation'].min():.4f}, {clean_df['correlation'].max():.4f}]")
    significant = (clean_df['p_value'] < 0.05).sum()
    print(f"    Significant (p < 0.05): {significant}/{len(clean_df)}")
    
    # Normality test (only if enough data)
    if len(clean_df) >= 3:
        try:
            stat, p_norm = shapiro(clean_df['correlation'])
            print(f"\n  Shapiro-Wilk Normality Test: W={stat:.4f}, p={p_norm:.4f}")
            if p_norm < 0.05:
                print("    → Data significantly deviates from normality")
            else:
                print("    → Data appears normally distributed")
        except:
            print("    → Could not compute normality test (too few data points)")
    
    # One-sample t-test
    if len(clean_df) >= 2:
        t_stat, p_val = ttest_1samp(clean_df['correlation'], 0)
        print(f"\n  One-sample t-test against 0:")
        print(f"    t = {t_stat:.4f}, p = {p_val:.4f}")
        if p_val < 0.05:
            print("    → Significant: Mean correlation differs from 0")
        else:
            print("    → Not significant: Mean correlation does not differ from 0")


### step 7b: results and statistics averaged across dimensions ###

df_avg = pd.DataFrame(results_averaged)
dfs_by_dimension['averaged'] = df_avg   # so it's included in later plotting if needed

clean_avg = df_avg.dropna(subset=['correlation'])
n_removed_avg = len(df_avg) - len(clean_avg)

print(f"\n{'='*40}")
print(f"DIMENSION: AVERAGED (across all 5)")
print(f"{'='*40}")

if len(clean_avg) == 0:
    print(f"  No valid correlations (all NaN)")
    print(f"  Total artworks: {len(df_avg)}")
    print(f"  With NaN: {n_removed_avg}")
else:
    print(f"  Total artworks: {len(df_avg)}")
    if n_removed_avg > 0:
        print(f"  Removed {n_removed_avg} artworks with NaN correlations")
    
    print(f"\n  Summary Statistics:")
    print(f"    Mean correlation: {clean_avg['correlation'].mean():.4f}")
    print(f"    Std deviation: {clean_avg['correlation'].std():.4f}")
    print(f"    Range: [{clean_avg['correlation'].min():.4f}, {clean_avg['correlation'].max():.4f}]")
    significant = (clean_avg['p_value'] < 0.05).sum()
    print(f"    Significant (p < 0.05): {significant}/{len(clean_avg)}")
    
    # Normality test
    if len(clean_avg) >= 3:
        try:
            stat, p_norm = shapiro(clean_avg['correlation'])
            print(f"\n  Shapiro-Wilk Normality Test: W={stat:.4f}, p={p_norm:.4f}")
            if p_norm < 0.05:
                print("    → Data significantly deviates from normality")
            else:
                print("    → Data appears normally distributed")
        except:
            print("    → Could not compute normality test (too few data points)")
    
    # One-sample t-test
    if len(clean_avg) >= 2:
        t_stat, p_val = ttest_1samp(clean_avg['correlation'], 0)
        print(f"\n  One-sample t-test against 0:")
        print(f"    t = {t_stat:.4f}, p = {p_val:.4f}")
        if p_val < 0.05:
            print("    → Significant: Mean correlation differs from 0")
        else:
            print("    → Not significant: Mean correlation does not differ from 0")


### combined swarm plot - all 4 metrics ###

swarm_data = []

# Panels 1-3: Fixation, Blink, TTFF (from corr_df)
for _, row in corr_df.iterrows():
    swarm_data.append({
        'metric': 'Fixation Count',
        'correlation': row['corr_fixation']
    })
    swarm_data.append({
        'metric': 'Blink Count',
        'correlation': row['corr_blink']
    })
    swarm_data.append({
        'metric': 'Time to First Fixation',
        'correlation': row['corr_ttff']
    })

# Panel 4: Averaged Scanpath (from clean_avg, already filtered for NaNs)
for _, row in clean_avg.iterrows():
    swarm_data.append({
        'metric': 'Scanpath Similarity',
        'correlation': row['correlation']
    })

swarm_df = pd.DataFrame(swarm_data)

# Create swarm plot
plt.figure(figsize=(12, 6))

sns.swarmplot(
    data=swarm_df,
    x='metric',
    y='correlation',
    palette=['blue', 'green', 'red', 'purple'],
    size=8,
    alpha=0.7,
    edgecolor='black',
    linewidth=0.5
)

plt.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)

plt.xlabel('Eye Movement Metric', fontsize=12)
plt.ylabel('RDM Correlation', fontsize=12)
plt.title('Distribution of RDM Correlations Across Artworks', fontsize=14)

plt.ylim(-1, 1)
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# Print summary statistics for reference
print("\nSummary statistics for swarm plot data:")
print(swarm_df.groupby('metric')['correlation'].agg(['mean', 'std', 'min', 'max', 'count']))


### Add Cohen's d 

# 2a - 2c
import numpy as np
from scipy.stats import ttest_1samp

print("\nOne-sample t-test against 0 (Z-scored) with Cohen's d")
for metric in ["fixation", "blink", "ttff"]:
    corr_col = f"corr_{metric}"
    data = corr_df[corr_col].dropna() # Good practice to drop NaNs just in case
    
    # 1. Run the t-test
    t_stat, p_val = ttest_1samp(data, 0)
    
    # 2. Calculate Cohen's d
    mean_corr = data.mean()
    sd_corr = data.std() # pandas uses ddof=1 by default, which is correct here
    d_val = mean_corr / sd_corr
    
    print(f"{metric.upper()}: t={t_stat:.4f}, p={p_val:.4f}, d={d_val:.4f}")


# 2d (scanpaths)
clean_avg = df_avg.dropna(subset=['correlation'])
data_avg = clean_avg['correlation']

t_stat_avg, p_val_avg = ttest_1samp(data_avg, 0)
d_avg = data_avg.mean() / data_avg.std()

print(f"SCANPATH AVERAGED: t={t_stat_avg:.4f}, p={p_val_avg:.4f}, d={d_avg:.4f}")

print("\n" + "="*60)
print("EFFECT SIZES (COHEN'S D) FOR INDIVIDUAL SCANPATH DIMENSIONS")
print("="*60)

for dim_name in MM_DIMENSIONS:
    df = dfs_by_dimension[dim_name]
    
    # Drop NaNs to get valid correlations
    clean_df = df.dropna(subset=['correlation'])
    data = clean_df['correlation']
    
    if len(data) >= 2:
        # Calculate t-test (to get df and p-value if you want to print them again)
        t_stat, p_val = ttest_1samp(data, 0)
        
        # Calculate Cohen's d
        mean_corr = data.mean()
        sd_corr = data.std()
        d_val = mean_corr / sd_corr
        
        print(f"{dim_name.upper():<10}: t={t_stat:.3f}, p={p_val:.3f}, d={d_val:.3f} (n={len(data)})")
    else:
        print(f"{dim_name.upper():<10}: Not enough data to calculate effect size.")