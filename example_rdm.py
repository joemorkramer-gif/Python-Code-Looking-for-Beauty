import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform, pdist
from scipy.stats import zscore
import glob
import os

# ============================================
# CONFIGURATION
# ============================================
BEHAVIORAL_DATA_PATH = r"/Users/joemorkramer/Documents/Uni/SoSe 26/Seafile/Beauty and eyes/behavioral summary"

# ============================================
# LOAD DATA
# ============================================
os.chdir(BEHAVIORAL_DATA_PATH)
csv_files = glob.glob("*.csv")
print(f"Found {len(csv_files)} CSV files")

all_data = []
for file in csv_files:
    df = pd.read_csv(file)
    df['participant'] = os.path.splitext(file)[0].replace("_summary", "")
    all_data.append(df)

combined_df = pd.concat(all_data, ignore_index=True)
artwork_ids = sorted(combined_df['artwork_id'].unique())
print(f"Found {len(artwork_ids)} artworks")

# ============================================
# PICK AN ARTWORK
# ============================================
artwork_id = artwork_ids[0]   # Change to any artwork index

artwork_df = combined_df[combined_df['artwork_id'] == artwork_id].copy()
artwork_df['rating_z'] = zscore(artwork_df['rating'])
artwork_df = artwork_df.sort_values('participant').reset_index(drop=True)

ratings = artwork_df['rating_z'].values[:, np.newaxis]
n_subjects = len(ratings)

print(f"\nArtwork: {artwork_id}")
print(f"Number of subjects: {n_subjects}")

# ============================================
# COMPUTE RDM
# ============================================
rdm_condensed = pdist(ratings, metric=lambda u, v: np.abs(u[0] - v[0]))
rdm = squareform(rdm_condensed)

# ============================================
# PLOT
# ============================================
fig, ax = plt.subplots(figsize=(7, 7))

cmap = plt.cm.coolwarm
im = ax.imshow(rdm, cmap=cmap, interpolation='nearest', aspect='equal')

# Diagonal line
ax.plot([-0.5, n_subjects - 0.5], [-0.5, n_subjects - 0.5],
        color='black', linewidth=1, alpha=0.5)

# Axis labels
ax.set_xlabel('Participants', fontsize=12)
ax.set_ylabel('Participants', fontsize=12)

# ---- 1-based tick labels ----
ticks = np.arange(0, n_subjects, 5)
if ticks[-1] != n_subjects - 1:
    ticks = np.append(ticks, n_subjects - 1)

ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(ticks + 1)
ax.set_yticklabels(ticks + 1)
ax.tick_params(labelsize=10)

# Title
ax.set_title('Rating RDM', fontsize=14, fontweight='bold', loc='left', pad=10)

# Colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Absolute Difference\nin Z-Scored Ratings', fontsize=10)
cbar.ax.tick_params(labelsize=9)

plt.tight_layout()
# plt.savefig('exemplary_rating_rdm.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.show()
