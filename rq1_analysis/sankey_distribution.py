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

total_w = df['SPFWT0'].sum()

# -----------------------------------------------------------------------------
# 2. COMPUTE NODE LOCATIONS AND WEIGHTS
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(22, 14))

# Define levels
# Y: 4(Total), 3(Track), 2(Gender), 1(SES), 0(Migration)

# Level 0: Root
root_x, root_y = 0, 4
ax.text(root_x, root_y, f"All Graduates\nN=1,828", ha='center', va='center', 
        bbox=dict(boxstyle="round,pad=0.6", facecolor='#404040', edgecolor='black', linewidth=1.5),
        color='white', fontsize=13, weight='bold')

# Track allocations
acad_w = df[df['ED_GROUP'] == 0.0]['SPFWT0'].sum()
voc_w = df[df['ED_GROUP'] == 1.0]['SPFWT0'].sum()

track_coords = {0.0: (-6, 3), 1.0: (6, 3)}
track_labels = {0.0: "Academic\nN=1,441", 1.0: "Vocational\nN=387"}
track_colors = {0.0: '#1b9e77', 1.0: '#d95f02'}

# Connect Root -> Track
for t_val, (tx, ty) in track_coords.items():
    w = acad_w if t_val == 0.0 else voc_w
    l_width = (w / total_w) * 40
    ax.plot([root_x, tx], [root_y - 0.2, ty + 0.2], color=track_colors[t_val], alpha=0.4, linewidth=max(1, l_width))
    ax.text(tx, ty, track_labels[t_val], ha='center', va='center', 
            bbox=dict(boxstyle="round,pad=0.5", facecolor=track_colors[t_val], edgecolor='black'),
            color='white', fontsize=12, weight='bold')

# Track -> Gender
gender_coords = {}
for t_val, (tx, ty) in track_coords.items():
    subset_t = df[df['ED_GROUP'] == t_val]
    g_offsets = [-2.5, 2.5] if t_val == 0.0 else [-1.5, 1.5]
    
    for g_idx, (g_val, g_lbl) in enumerate({1.0: "Male", 2.0: "Female"}.items()):
        subset_g = subset_t[subset_t['GENDER_R'] == g_val]
        w = subset_g['SPFWT0'].sum()
        n = len(subset_g)
        
        gx, gy = tx + g_offsets[g_idx], ty - 1
        gender_coords[(t_val, g_val)] = (gx, gy)
        
        l_width = (w / total_w) * 40
        ax.plot([tx, gx], [ty - 0.2, gy + 0.2], color=track_colors[t_val], alpha=0.4, linewidth=max(1, l_width))
        
        c_gen = '#168060' if t_val == 0.0 else '#a14600'
        ax.text(gx, gy, f"{g_lbl}\nN={n}", ha='center', va='center', 
                bbox=dict(boxstyle="round,pad=0.4", facecolor=c_gen, edgecolor='black'),
                color='white', fontsize=11, weight='bold')

# Gender -> SES
ses_coords = {}
for (t_val, g_val), (gx, gy) in gender_coords.items():
    subset_g = df[(df['ED_GROUP'] == t_val) & (df['GENDER_R'] == g_val)]
    s_offsets = [-0.8, 0.0, 0.8]
    
    for s_idx, (s_val, s_lbl) in enumerate({1.0: "Low", 2.0: "Med", 3.0: "High"}.items()):
        subset_s = subset_g[subset_g['PAREDC2'] == s_val]
        w = subset_s['SPFWT0'].sum()
        n = len(subset_s)
        
        sx, sy = gx + s_offsets[s_idx], gy - 1
        ses_coords[(t_val, g_val, s_val)] = (sx, sy)
        
        if n > 0:
            l_width = (w / total_w) * 40
            ax.plot([gx, sx], [gy - 0.2, sy + 0.2], color=track_colors[t_val], alpha=0.4, linewidth=max(1, l_width))
            
            c_ses = '#36c49a' if t_val == 0.0 else '#ff8833'
            ax.text(sx, sy, f"{s_lbl}\nN={n}", ha='center', va='center', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=c_ses, edgecolor='black'),
                    color='white', fontsize=10)

# SES -> Migration
for (t_val, g_val, s_val), (sx, sy) in ses_coords.items():
    subset_s = df[(df['ED_GROUP'] == t_val) & (df['GENDER_R'] == g_val) & (df['PAREDC2'] == s_val)]
    m_offsets = [-0.25, 0.25]
    
    for m_idx, (m_val, m_lbl) in enumerate({1.0: "Nat", 2.0: "For"}.items()):
        subset_m = subset_s[subset_s['A2_Q03a_T'] == m_val]
        w = subset_m['SPFWT0'].sum()
        n = len(subset_m)
        
        mx, my = sx + m_offsets[m_idx], sy - 1
        
        if n > 0:
            l_width = (w / total_w) * 40
            ax.plot([sx, mx], [sy - 0.2, my + 0.2], color=track_colors[t_val], alpha=0.4, linewidth=max(1, l_width))
            
            c_mig = '#26c294' if t_val == 0.0 else '#ffb366'
            ax.text(mx, my, f"{m_lbl}\nN={n}", ha='center', va='center', 
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=c_mig, edgecolor='none'),
                    color='black', fontsize=8)

# -----------------------------------------------------------------------------
# 3. FINAL STYLING AND EXPORT
# -----------------------------------------------------------------------------
ax.set_xlim(-12, 12)
ax.set_ylim(-0.5, 4.5)
ax.axis('off')

# Level Y-axis descriptions on the far left
ax.text(-11, 4, "Root Pool", fontsize=14, ha='left', weight='bold', color='#333')
ax.text(-11, 3, "Track Selection", fontsize=14, ha='left', weight='bold', color='#333')
ax.text(-11, 2, "Gender Distribution", fontsize=14, ha='left', weight='bold', color='#333')
ax.text(-11, 1, "Parental SES", fontsize=14, ha='left', weight='bold', color='#333')
ax.text(-11, 0, "Migration Status", fontsize=14, ha='left', weight='bold', color='#333')

ax.set_title("Static Top-to-Bottom Pathway Flow (Thesis Ready)", fontsize=16, weight='bold', pad=20)

plot_path = os.path.join(output_dir, 'rq1_sankey_intersectional.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Success. Static hierarchical tree flow saved: {plot_path}")
