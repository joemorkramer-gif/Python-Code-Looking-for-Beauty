import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os
import glob
import multimatch_gaze as mmg

# ============================================
# CONFIGURATION
# ============================================
SCANPATH_PATH = "/Users/joemorkramer/Documents/Uni/SoSe 26/Seafile/Beauty and eyes/scanpath"
OUTPUT_DIR = "/Users/joemorkramer/Documents/Uni/SoSe 26/Seafile/Beauty and eyes/scanpath_comparisons"
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800

# Your artwork ID
ARTWORK_ID = "0a337ba2-fa57-176e-3cd6-a8ea27d41ac8"

# ============================================
# SPECIFY YOUR PARTICIPANTS HERE
# ============================================
PARTICIPANT1 = "sub-EA102"  # <--- CHANGE THIS to your first participant
PARTICIPANT2 = "sub-EA125"  # <--- CHANGE THIS to your second participant

# ============================================
# FUNCTION TO CONVERT DATAFRAME TO RECORD ARRAY
# ============================================
def df_to_recarray(df):
    """Convert pandas DataFrame to structured array for MultiMatch"""
    return np.rec.fromarrays(
        (df['x'].values.astype(float),
         df['y'].values.astype(float),
         df['duration [ms]'].values.astype(float) / 1000),
        dtype=[('start_x', 'f8'), ('start_y', 'f8'), ('duration', 'f8')]
    )

# ============================================
# FUNCTION TO FIND SCANPATH FILE
# ============================================
def find_scanpath_file(participant_folder, artwork_id):
    """Find a scanpath file for a given participant and artwork."""
    all_files = glob.glob(os.path.join(participant_folder, "*.csv"))
    all_files.extend(glob.glob(os.path.join(participant_folder, "*.tsv")))
    
    # Try exact match
    for file_path in all_files:
        file_name = os.path.basename(file_path)
        file_name_no_ext = os.path.splitext(file_name)[0]
        if file_name_no_ext == artwork_id:
            return file_path
    
    # Try partial match
    for file_path in all_files:
        file_name = os.path.basename(file_path)
        if artwork_id in file_name:
            return file_path
    
    return None

# ============================================
# FUNCTION TO PLOT SCANPATHS ONLY
# ============================================
def plot_scanpaths_only(participant1_file, participant2_file, 
                        participant1_id, participant2_id,
                        screen_width=SCREEN_WIDTH, 
                        screen_height=SCREEN_HEIGHT,
                        save_path=None):
    """
    Plot two scanpaths side by side without statistics.
    """
    
    # Read fixation data
    df1 = pd.read_csv(participant1_file)
    df2 = pd.read_csv(participant2_file)
    
    # Create figure with two side-by-side panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- LEFT PANEL: Participant 1 ---
    ax1.set_facecolor('white')
    ax1.set_xlim(0, screen_width)
    ax1.set_ylim(screen_height, 0)
    ax1.set_title(f'Participant 1\n{len(df1)} fixations', fontsize=14, fontweight='bold')
    
    # Plot saccades for P1
    if len(df1) > 1:
        for i in range(len(df1) - 1):
            ax1.plot([df1.iloc[i]['x'], df1.iloc[i+1]['x']], 
                    [df1.iloc[i]['y'], df1.iloc[i+1]['y']], 
                    'blue', alpha=0.5, linewidth=1.5, linestyle='-')
    
    # Plot fixations for P1
    for idx, fixation in df1.iterrows():
        duration = fixation['duration [ms]']
        radius = 12 + (duration / 200)
        
        circle = Circle((fixation['x'], fixation['y']), 
                       radius=radius,
                       facecolor='blue', 
                       edgecolor='darkblue',
                       alpha=0.5,
                       linewidth=1.5)
        ax1.add_patch(circle)
        
        # Fixation number
        ax1.text(fixation['x'], fixation['y'] - radius - 8, 
                str(idx + 1), ha='center', va='top', 
                fontsize=10, fontweight='bold', color='black')
    
    ax1.set_xlabel('X Position (pixels)', fontsize=12)
    ax1.set_ylabel('Y Position (pixels)', fontsize=12)
    ax1.grid(True, alpha=0.2)
    
    # --- RIGHT PANEL: Participant 2 ---
    ax2.set_facecolor('white')
    ax2.set_xlim(0, screen_width)
    ax2.set_ylim(screen_height, 0)
    ax2.set_title(f'Participant 2\n{len(df2)} fixations', fontsize=14, fontweight='bold')
    
    # Plot saccades for P2
    if len(df2) > 1:
        for i in range(len(df2) - 1):
            ax2.plot([df2.iloc[i]['x'], df2.iloc[i+1]['x']], 
                    [df2.iloc[i]['y'], df2.iloc[i+1]['y']], 
                    'red', alpha=0.5, linewidth=1.5, linestyle='-')
    
    # Plot fixations for P2
    for idx, fixation in df2.iterrows():
        duration = fixation['duration [ms]']
        radius = 12 + (duration / 200)
        
        circle = Circle((fixation['x'], fixation['y']), 
                       radius=radius,
                       facecolor='red', 
                       edgecolor='darkred',
                       alpha=0.5,
                       linewidth=1.5)
        ax2.add_patch(circle)
        
        # Fixation number
        ax2.text(fixation['x'], fixation['y'] - radius - 8, 
                str(idx + 1), ha='center', va='top', 
                fontsize=10, fontweight='bold', color='black')
    
    ax2.set_xlabel('X Position (pixels)', fontsize=12)
    ax2.set_ylabel('Y Position (pixels)', fontsize=12)
    ax2.grid(True, alpha=0.2)
    
    # Main title
    fig.suptitle('Exemplary Scanpaths of Two Participants', fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to: {save_path}")
    
    return fig, (ax1, ax2)

# ============================================
# FUNCTION TO COMPARE AND PRINT STATISTICS
# ============================================
def print_multimatch_stats(file1, file2, participant1_id, participant2_id):
    """Calculate and print MultiMatch statistics to terminal."""
    
    # Read data
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    # Convert to record arrays
    vec1 = df_to_recarray(df1)
    vec2 = df_to_recarray(df2)
    
    # Calculate MultiMatch scores
    try:
        scores = mmg.docomparison(
            vec1, vec2,
            screensize=[SCREEN_WIDTH, SCREEN_HEIGHT],
            grouping=False
        )
        dim_names = ['Vector', 'Direction', 'Length', 'Position', 'Duration']
        score_dict = dict(zip(dim_names, scores))
        avg_score = np.mean(scores)
        
        # Print statistics
        print("\n" + "="*60)
        print("MULTIMATCH SIMILARITY STATISTICS")
        print("="*60)
        print(f"\nComparing: {participant1_id} vs {participant2_id}")
        print(f"\nIndividual Dimension Scores:")
        for dim, score in score_dict.items():
            print(f"  {dim:10s}: {score:.4f}")
        print(f"\n{'Average Similarity':20s}: {avg_score:.4f}")
        
        # Interpretation
        print("\nInterpretation:")
        if avg_score >= 0.8:
            print("  → High similarity - Very similar scanpaths")
        elif avg_score >= 0.6:
            print("  → Moderate similarity - Somewhat similar scanpaths")
        elif avg_score >= 0.4:
            print("  → Low similarity - Quite different scanpaths")
        else:
            print("  → Very low similarity - Very different scanpaths")
        print("="*60)
        
        return score_dict, avg_score
        
    except Exception as e:
        print(f"Error calculating MultiMatch scores: {e}")
        return None, None

# ============================================
# MAIN EXECUTION
# ============================================
if __name__ == "__main__":
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*60)
    print(f"SCANPATH COMPARISON: {PARTICIPANT1} vs {PARTICIPANT2}")
    print(f"Artwork: {ARTWORK_ID}")
    print("="*60)
    
    # Build participant folder paths
    folder1 = os.path.join(SCANPATH_PATH, PARTICIPANT1)
    folder2 = os.path.join(SCANPATH_PATH, PARTICIPANT2)
    
    # Find the scanpath files
    file1 = find_scanpath_file(folder1, ARTWORK_ID)
    file2 = find_scanpath_file(folder2, ARTWORK_ID)
    
    if file1 is None:
        print(f"❌ File not found for {PARTICIPANT1}")
        print(f"   Looking in: {folder1}")
        print(f"   Available files: {os.listdir(folder1)[:5]}...")
        exit()
    if file2 is None:
        print(f"❌ File not found for {PARTICIPANT2}")
        print(f"   Looking in: {folder2}")
        print(f"   Available files: {os.listdir(folder2)[:5]}...")
        exit()
    
    print(f"\n✅ Found files:")
    print(f"   {PARTICIPANT1}: {os.path.basename(file1)}")
    print(f"   {PARTICIPANT2}: {os.path.basename(file2)}")
    
    # Print MultiMatch statistics to terminal
    print_multimatch_stats(file1, file2, PARTICIPANT1, PARTICIPANT2)
    
    # Generate and save the figure (scanpaths only)
    short_id = ARTWORK_ID[:8]
    save_path = os.path.join(OUTPUT_DIR, f"scanpath_comparison_{PARTICIPANT1}_{PARTICIPANT2}_{short_id}.png")
    
    fig, axes = plot_scanpaths_only(
        file1, file2,
        PARTICIPANT1, PARTICIPANT2,
        save_path=save_path
    )
    
    plt.show()
    
    print(f"\n✅ Figure saved to: {save_path}")