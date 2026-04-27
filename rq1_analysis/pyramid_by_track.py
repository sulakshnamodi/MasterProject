import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

# -----------------------------------------------------------------------------
# 1. SETUP AND DATA LOADING
# -----------------------------------------------------------------------------
subdataset_root = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\data\preprocessed\subdataset1'
subdataset_filepath = os.path.join(subdataset_root, 'piaac_norway_subdataset1.pkl')
output_dir = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\rq1_analysis'

os.makedirs(output_dir, exist_ok=True)

with open(subdataset_filepath, 'rb') as f:
    loaded_data = pickle.load(f)

df = loaded_data['dataframe'].copy()
anchors = ['ED_GROUP', 'GENDER_R', 'PAREDC2', 'A2_Q03a_T']
df = df.dropna(subset=anchors + ['SPFWT0']).copy()

# -----------------------------------------------------------------------------
# 2. DEFINE PLOTTING FUNCTION
# -----------------------------------------------------------------------------
def generate_track_pyramid(track_val, track_name, track_color, filename):
    df_track = df[df['ED_GROUP'] == track_val].copy()
    track_w = df_track['SPFWT0'].sum()
    
    # Layer 0: Track Root
    l0_fracs, l0_ns = [1.0], [len(df_track)]
    l0_keys = [()]
    l0_colors = [track_color]
    
    # Layer 1: Gender
    l1_fracs, l1_ns = [], []
    l1_keys = []
    l1_colors = []
    for g_val in [1.0, 2.0]:
        sub = df_track[df_track['GENDER_R'] == g_val]
        l1_fracs.append(sub['SPFWT0'].sum() / track_w)
        l1_ns.append(len(sub))
        l1_keys.append((g_val,))
        l1_colors.append('#457b9d' if g_val == 1.0 else '#e07a5f')
        
    # Layer 2: SES
    l2_fracs, l2_ns = [], []
    l2_keys = []
    l2_colors = []
    ses_cmap = {1.0: '#f4f1de', 2.0: '#e07a5f', 3.0: '#3d405b'}
    for g_val in [1.0, 2.0]:
        for s_val in [1.0, 2.0, 3.0]:
            sub = df_track[(df_track['GENDER_R'] == g_val) & (df_track['PAREDC2'] == s_val)]
            l2_fracs.append(sub['SPFWT0'].sum() / track_w)
            l2_ns.append(len(sub))
            l2_keys.append((g_val, s_val))
            l2_colors.append(ses_cmap[s_val])
            
    # Layer 3: Migration
    l3_fracs, l3_ns = [], []
    l3_keys = []
    l3_colors = []
    mig_cmap = {1.0: '#81b29a', 2.0: '#f2cc8f'}
    for g_val in [1.0, 2.0]:
        for s_val in [1.0, 2.0, 3.0]:
            for m_val in [1.0, 2.0]:
                sub = df_track[(df_track['GENDER_R'] == g_val) & (df_track['PAREDC2'] == s_val) & (df_track['A2_Q03a_T'] == m_val)]
                l3_fracs.append(sub['SPFWT0'].sum() / track_w)
                l3_ns.append(len(sub))
                l3_keys.append((g_val, s_val, m_val))
                l3_colors.append(mig_cmap[m_val])

    # -------------------------------------------------------------------------
    # RENDERING
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(15, 9))
    
    layers = [
        (l0_fracs, l0_ns, l0_keys, l0_colors, 30, 0.0, "Track Pool"),
        (l1_fracs, l1_ns, l1_keys, l1_colors, 50, 0.8, "Gender"),
        (l2_fracs, l2_ns, l2_keys, l2_colors, 70, 1.2, "Parental SES"),
        (l3_fracs, l3_ns, l3_keys, l3_colors, 90, 1.5, "Migration Status")
    ]
    
    box_centers = {}
    y_coords = [2.4, 1.6, 0.8, 0]
    
    for idx, (fracs, ns, keys, colors, total_width, gap_w, title) in enumerate(layers):
        y_val = y_coords[idx]
        n_segments = len(fracs)
        
        total_gaps = (n_segments - 1) * gap_w if n_segments > 1 else 0
        usable_width = total_width - total_gaps
        
        start_x = -total_width / 2
        leftmost_edge = start_x
        
        for f, n, k, c in zip(fracs, ns, keys, colors):
            w = f * usable_width
            if w > 0.01:
                ax.barh(y_val, w, left=start_x, color=c, edgecolor='white', height=0.65, linewidth=1.0, zorder=3)
                
                center_x = start_x + w / 2
                box_centers[k] = (center_x, y_val)
                
                txt_color = 'black' if c in ['#f4f1de', '#f2cc8f'] else 'white'
                
                if f > 0.015:
                    if idx == 0:
                        lbl = f"N={n:,}\n{f*100:.1f}%"
                    else:
                        lbl = f"{f*100:.1f}%"
                        
                    ax.text(center_x, y_val, lbl, ha='center', va='center', 
                            color=txt_color, fontsize=max(7, 11 - idx), weight='bold', zorder=4)
                    
                start_x += w + gap_w
                
        ax.text(leftmost_edge - 3, y_val, title, ha='right', va='center', fontsize=14, weight='bold', color='#262626')

    # CONNECTORS
    for child_key, child_pos in box_centers.items():
        if len(child_key) > 0:
            parent_key = child_key[:-1]
            if parent_key in box_centers:
                parent_pos = box_centers[parent_key]
                ax.annotate(
                    "", 
                    xy=(child_pos[0], child_pos[1] + 0.32), 
                    xytext=(parent_pos[0], parent_pos[1] - 0.32),
                    arrowprops=dict(arrowstyle="->", color='#333333', lw=1.0, alpha=0.7),
                    zorder=1
                )
                
    # LEGEND
    import matplotlib.patches as mpatches
    legend_patches = [
        mpatches.Patch(color=track_color, label=f'{track_name} Track'),
        mpatches.Patch(color='#457b9d', label='Male'),
        mpatches.Patch(color='#e07a5f', label='Female'),
        mpatches.Patch(color='#f4f1de', label='Low SES'),
        mpatches.Patch(color='#e07a5f', label='Med SES'),
        mpatches.Patch(color='#3d405b', label='High SES'),
        mpatches.Patch(color='#81b29a', label='Native-Born'),
        mpatches.Patch(color='#f2cc8f', label='Foreign-Born')
    ]
    ax.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=11, frameon=False)
    
    ax.set_xlim(-65, 55)
    ax.set_ylim(-0.8, 3.0)
    ax.axis('off')
    
    plt.suptitle(f"Hierarchical Flow: {track_name} Sample Pathway", fontsize=17, weight='bold', y=0.96)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")

# -----------------------------------------------------------------------------
# 3. EXECUTE BOTH TRACKS
# -----------------------------------------------------------------------------
generate_track_pyramid(0.0, "Academic", "#1b9e77", "rq1_academic_pyramid.png")
generate_track_pyramid(1.0, "Vocational", "#d95f02", "rq1_vocational_pyramid.png")
