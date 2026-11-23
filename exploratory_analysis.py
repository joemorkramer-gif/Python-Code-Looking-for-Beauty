import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import glob
import os

### Configuration ###
BEHAVIORAL_DATA_PATH = r"/Users/joemorkramer/Documents/Uni/SoSe 26/Seafile/Beauty and eyes/behavioral summary"
SCANPATH_DATA_PATH = r"/Users/joemorkramer/Documents/Uni/SoSe 26/Seafile/Beauty and eyes/scanpath"

# Define 4 time bins (in milliseconds)
TIME_BINS = [(0, 1000), (1000, 2000), (2000, 3000), (3000, 4000)]
BIN_LABELS = ["0-1s", "1-2s", "2-3s", "3-4s"]


# Step 1: Average ratings per artwork 
#============================================
os.chdir(BEHAVIORAL_DATA_PATH)
behavioral_files = glob.glob("*.csv")

all_ratings = []
for file in behavioral_files:
    df = pd.read_csv(file)
    # Keep only artwork_id and rating
    all_ratings.append(df[['artwork_id', 'rating']])

# Combine all participants into one DataFrame
ratings_df = pd.concat(all_ratings, ignore_index=True)

# Calculate the average rating for each artwork across all participants
avg_ratings = ratings_df.groupby('artwork_id')['rating'].mean().reset_index()
avg_ratings.rename(columns={'rating': 'average_rating'}, inplace=True)

print(f"Calculated average ratings for {len(avg_ratings)} artworks.")


# Step 2: Average fixations per time bin 
# ============================================
scanpath_participant_ids = [os.path.basename(f) for f in glob.glob(os.path.join(SCANPATH_DATA_PATH, "sub-EA*"))]

# Store every participant's binned fixations in a list
all_binned_data = []

for participant_id in scanpath_participant_ids:
    participant_folder = os.path.join(SCANPATH_DATA_PATH, participant_id)
    scanpath_files = glob.glob(os.path.join(participant_folder, "*.csv"))
    scanpath_files.extend(glob.glob(os.path.join(participant_folder, "*.tsv")))
    
    for file_path in scanpath_files:
        file_name = os.path.basename(file_path)
        artwork_id = os.path.splitext(file_name)[0]
        
        # Clean artwork ID (same as previous code)
        for prefix in ["artwork_", "img_", "fix_", "image_", "stim_", "scanpath_"]:
            if artwork_id.startswith(prefix):
                artwork_id = artwork_id[len(prefix):]
                
        try:
            if file_path.endswith('.tsv'):
                df = pd.read_csv(file_path, sep='\t')
            else:
                df = pd.read_csv(file_path)
                
            if len(df) < 2 or 'duration [ms]' not in df.columns:
                continue
                
            # Calculate start time of each fixation
            df['start_time'] = df['duration [ms]'].cumsum().shift(1).fillna(0)
            
            # Count fixations in each of the 4 bins
            bin_counts = {}
            for (start, end), label in zip(TIME_BINS, BIN_LABELS):
                count = len(df[(df['start_time'] >= start) & (df['start_time'] < end)])
                bin_counts[label] = count
                
            # Store the data
            bin_counts['artwork_id'] = artwork_id
            bin_counts['participant'] = participant_id
            all_binned_data.append(bin_counts)
            
        except Exception as e:
            pass

# Convert to DataFrame
fixations_df = pd.DataFrame(all_binned_data)

# Calculate the average fixation count per artwork for each time bin
avg_fixations = fixations_df.groupby('artwork_id')[BIN_LABELS].mean().reset_index()

print(f"Calculated average fixations for {len(avg_fixations)} artworks.")


# Step 3: Correlations
# ============================================
# Merge the ratings and fixations DataFrames
merged_df = pd.merge(avg_ratings, avg_fixations, on='artwork_id', how='inner')
print(f"Successfully merged data for {len(merged_df)} artworks.\n")

print("="*50)
print("SPEARMAN CORRELATIONS: Average Rating vs. Average Fixations")
print("="*50)

results = []

for label in BIN_LABELS:
    # Drop NaNs in case an artwork had missing data for this bin
    clean_data = merged_df.dropna(subset=['average_rating', label])
    
    # Run Spearman correlation
    corr, p_val = spearmanr(clean_data['average_rating'], clean_data[label])
    
    print(f"Time Bin {label}:")
    print(f"  Spearman ρ = {corr:.4f}")
    print(f"  p-value    = {p_val:.4e}")
    print(f"  (n = {len(clean_data)} artworks)\n")
    
    results.append({
        'Time Bin': label,
        'Correlation': corr,
        'p-value': p_val
    })


# Step 4: Scatterplots
# ============================================
# Create a 1x4 subplot grid
fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
fig.suptitle('Correlation between Average Artwork Rating and Fixation Counts over Time', fontsize=16)

for i, label in enumerate(BIN_LABELS):
    #     # Scatter plot
    sns.scatterplot(
        data=merged_df, 
        x='average_rating', 
        y=label, 
        ax=axes[i],
        alpha=0.5, 
        color='gray', 
        s=25
    )
    
    # Get stats for the title
    res = results[i]
    axes[i].set_title(f"Time: {label}")
    axes[i].set_xlabel('Average Rating')
    
    if i == 0:
        axes[i].set_ylabel('Average Fixation Count')

plt.tight_layout()
plt.show()


# ============================================
# Step 5: Z-scored Data Procedure
# ============================================
from scipy.stats import zscore

print("\n" + "="*50)
print("PROCESSING Z-SCORED DATA")
print("="*50)

# --- 5a: Z-score Ratings ---
# We z-score within each participant's dataframe
zscored_ratings = []
for file in behavioral_files:
    df = pd.read_csv(file)
    # Create copy and z-score
    df_z = df[['artwork_id', 'rating']].copy()
    df_z['rating_z'] = zscore(df['rating'])
    zscored_ratings.append(df_z)

# Combine and average
ratings_z_df = pd.concat(zscored_ratings, ignore_index=True)
avg_ratings_z = ratings_z_df.groupby('artwork_id')['rating_z'].mean().reset_index()
avg_ratings_z.rename(columns={'rating_z': 'average_rating_z'}, inplace=True)


# --- 5b: Z-score Fixations per Time Bin ---
# We need to store all participant bin counts first, z-score them, then average
all_binned_data_z = []

for participant_id in scanpath_participant_ids:
    participant_folder = os.path.join(SCANPATH_DATA_PATH, participant_id)
    scanpath_files = glob.glob(os.path.join(participant_folder, "*.csv"))
    scanpath_files.extend(glob.glob(os.path.join(participant_folder, "*.tsv")))
    
    # Temporary storage for this specific participant
    participant_bins = []
    
    for file_path in scanpath_files:
        file_name = os.path.basename(file_path)
        artwork_id = os.path.splitext(file_name)[0]
        
        for prefix in ["artwork_", "img_", "fix_", "image_", "stim_", "scanpath_"]:
            if artwork_id.startswith(prefix):
                artwork_id = artwork_id[len(prefix):]
                
        try:
            if file_path.endswith('.tsv'):
                df = pd.read_csv(file_path, sep='\t')
            else:
                df = pd.read_csv(file_path)
                
            if len(df) < 2 or 'duration [ms]' not in df.columns:
                continue
                
            df['start_time'] = df['duration [ms]'].cumsum().shift(1).fillna(0)
            
            bin_counts = {'artwork_id': artwork_id, 'participant': participant_id}
            for (start, end), label in zip(TIME_BINS, BIN_LABELS):
                count = len(df[(df['start_time'] >= start) & (df['start_time'] < end)])
                bin_counts[label] = count
                
            participant_bins.append(bin_counts)
            
        except Exception as e:
            pass
            
    # If this participant has data, z-score their bin counts before adding to the master list
    if participant_bins:
        p_df = pd.DataFrame(participant_bins)
        for label in BIN_LABELS:
            p_df[f'{label}_z'] = zscore(p_df[label])
        all_binned_data_z.append(p_df)

# Combine all participants and average the z-scored values
fixations_z_df = pd.concat(all_binned_data_z, ignore_index=True)
z_cols = [f'{label}_z' for label in BIN_LABELS]
avg_fixations_z = fixations_z_df.groupby('artwork_id')[z_cols].mean().reset_index()


# --- 5c: Merge Z-scored Data ---
merged_df_z = pd.merge(avg_ratings_z, avg_fixations_z, on='artwork_id', how='inner')
print(f"Successfully merged z-scored data for {len(merged_df_z)} artworks.\n")


# --- 5d: Correlations on Z-scored Data ---
print("="*50)
print("SPEARMAN CORRELATIONS: Average Rating (Z) vs. Average Fixations (Z)")
print("="*50)

results_z = []

for label in BIN_LABELS:
    z_col = f'{label}_z'
    clean_data = merged_df_z.dropna(subset=['average_rating_z', z_col])
    
    corr, p_val = spearmanr(clean_data['average_rating_z'], clean_data[z_col])
    
    print(f"Time Bin {label}:")
    print(f"  Spearman ρ = {corr:.4f}")
    print(f"  p-value    = {p_val:.4e}")
    print(f"  (n = {len(clean_data)} artworks)\n")
    
    results_z.append({
        'Time Bin': label,
        'Correlation': corr,
        'p-value': p_val
    })


# --- 5e: Scatterplots for Z-scored Data ---
fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
fig.suptitle('Z-scored Correlation: Average Artwork Rating vs. Fixation Counts over Time', fontsize=16)

for i, label in enumerate(BIN_LABELS):
    z_col = f'{label}_z'
    
    sns.scatterplot(
        data=merged_df_z, 
        x='average_rating_z', 
        y=z_col, 
        ax=axes[i],
        alpha=0.5, 
        color='gray', 
        s=25
    )
    
    axes[i].set_title(f"Time: {label}")
    axes[i].set_xlabel('Z-scored Average Rating')
    
    if i == 0:
        axes[i].set_ylabel('Z-scored Fixation Count')

plt.tight_layout()
plt.show()

## FDR correction
from statsmodels.stats.multitest import multipletests

print("\n" + "="*50)
print("FDR CORRECTION (BENJAMINI-HOCHBERG)")
print("="*50)

# 1. Extract the raw p-values from your existing 'results' list
raw_p_values = [res['p-value'] for res in results]

# 2. Apply Benjamini-Hochberg FDR correction
rejected, p_values_corrected, _, _ = multipletests(
    raw_p_values, 
    alpha=0.05, 
    method='fdr_bh'
)

# 3. Print the results
for i, label in enumerate(BIN_LABELS):
    print(f"Time Bin {label}:")
    print(f"  Raw p-value:       {raw_p_values[i]:.4e}")
    print(f"  FDR-corrected p:   {p_values_corrected[i]:.4e}")
    
    if rejected[i]:
        print("  → Significant after FDR correction")
    else:
        print("  → Not significant after FDR correction")
    print()