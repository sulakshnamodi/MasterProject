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
anchors = ['ED_GROUP', 'GENDER_R', 'PAREDC2', 'A2_Q03a_T', 'IMPARC2']
df = df.dropna(subset=anchors + ['SPFWT0']).copy()

total_w = df['SPFWT0'].sum()

# -----------------------------------------------------------------------------
# 2. COMPUTE PROPORTIONS
# -----------------------------------------------------------------------------
l0_fracs = [1.0]
l0_colors = ['#2b2d42']
l0_keys = [()]

l1_fracs = []
l1_colors = ['#1b9e77', '#d95f02']
l1_keys = []
for t_val in [0.0, 1.0]:
    sub = df[df['ED_GROUP'] == t_val]
    l1_fracs.append(sub['SPFWT0'].sum() / total_w)
    l1_keys.append((t_val,))

l2_fracs = []
l2_colors = []
l2_keys = []
for t_val in [0.0, 1.0]:
    for g_val in [1.0, 2.0]:
        sub = df[(df['ED_GROUP'] == t_val) & (df['GENDER_R'] == g_val)]
        l2_fracs.append(sub['SPFWT0'].sum() / total_w)
        l2_colors.append('#457b9d' if g_val == 1.0 else '#e07a5f')
        l2_keys.append((t_val, g_val))

l3_fracs = []
l3_colors = []
l3_keys = []
ses_cmap = {1.0: '#f4f1de', 2.0: '#e07a5f', 3.0: '#3d405b'}
for t_val in [0.0, 1.0]:
    for g_val in [1.0, 2.0]:
        for s_val in [1.0, 2.0, 3.0]:
            sub = df[(df['ED_GROUP'] == t_val) & (df['GENDER_R'] == g_val) & (df['PAREDC2'] == s_val)]
            l3_fracs.append(sub['SPFWT0'].sum() / total_w)
            l3_colors.append(ses_cmap[s_val])
            l3_keys.append((t_val, g_val, s_val))

l4_fracs = []
l4_colors = []
l4_keys = []
mig_cmap = {1.0: '#81b29a', 2.0: '#f2cc8f'}
for t_val in [0.0, 1.0]:
    for g_val in [1.0, 2.0]:
        for s_val in [1.0, 2.0, 3.0]:
            for m_val in [1.0, 2.0]:
                sub = df[(df['ED_GROUP'] == t_val) & (df['GENDER_R'] == g_val) & (df['PAREDC2'] == s_val) & (df['A2_Q03a_T'] == m_val)]
                l4_fracs.append(sub['SPFWT0'].sum() / total_w)
                l4_colors.append(mig_cmap[m_val])
                l4_keys.append((t_val, g_val, s_val, m_val))

# -----------------------------------------------------------------------------
# 3. PLOT
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(16, 10))

layers = [
    (l0_fracs, l0_keys, l0_colors, 30, 0.0, "Total Pool"),
    (l1_fracs, l1_keys, l1_colors, 50, 0.6, "Education Track"),
    (l2_fracs, l2_keys, l2_colors, 70, 1.0, "Gender"),
    (l3_fracs, l3_keys, l3_colors, 90, 1.2, "Parental SES"),
    (l4_fracs, l4_keys, l4_colors, 110, 1.5, "Migration Status")
]

box_centers = {}
y_coords = [3.2, 2.4, 1.6, 0.8, 0]

for idx, (fracs, keys, colors, total_width, gap_w, title) in enumerate(layers):
    y_val = y_coords[idx]
    n_segments = len(fracs)
    
    total_gaps = (n_segments - 1) * gap_w if n_segments > 1 else 0
    usable_width = total_width - total_gaps
    
    start_x = -total_width / 2
    leftmost_edge = start_x
    
    for f, k, c in zip(fracs, keys, colors):
        w = f * usable_width
        if w > 0.01:
            ax.barh(y_val, w, left=start_x, color=c, edgecolor='white', height=0.65, linewidth=1.0, zorder=3)
            
            center_x = start_x + w / 2
            box_centers[k] = (center_x, y_val)
            
            txt_color = 'black' if c in ['#f4f1de', '#f2cc8f'] else 'white'
            
            if f > 0.015:
                font_sz = max(7, 11 - idx)
                ax.text(center_x, y_val, f"{f*100:.1f}%", ha='center', va='center', 
                        color=txt_color, fontsize=font_sz, weight='bold', zorder=4)
                
            start_x += w + gap_w
            
    # Shift label closer to the left edge of the pyramid row
    ax.text(leftmost_edge - 3, y_val, title, ha='right', va='center', fontsize=15, weight='bold', color='#262626')

# -----------------------------------------------------------------------------
# 4. DRAW DARKER ARROWS
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# 5. LEGEND
# -----------------------------------------------------------------------------
import matplotlib.patches as mpatches
legend_patches = [
    mpatches.Patch(color='#1b9e77', label='Academic Track'),
    mpatches.Patch(color='#d95f02', label='Vocational Track'),
    mpatches.Patch(color='#457b9d', label='Male'),
    mpatches.Patch(color='#e07a5f', label='Female'),
    mpatches.Patch(color='#f4f1de', label='Low SES'),
    mpatches.Patch(color='#e07a5f', label='Med SES'),
    mpatches.Patch(color='#3d405b', label='High SES'),
    mpatches.Patch(color='#81b29a', label='Native-Born'),
    mpatches.Patch(color='#f2cc8f', label='Foreign-Born')
]
ax.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=5, fontsize=11, frameon=False)

ax.set_xlim(-75, 65)
ax.set_ylim(-0.8, 3.8)
ax.axis('off')

plt.suptitle("Hierarchical Population Flow Architecture", fontsize=18, weight='bold', y=0.96)
plt.tight_layout()

plot_path = os.path.join(output_dir, 'rq1_demographic_pyramid.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

print("Pyramid alignments final.")
