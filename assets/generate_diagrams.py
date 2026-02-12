
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path

def setup_style():
    plt.style.use('dark_background')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

def create_vector_diagram(output_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define vectors
    origin = np.array([0, 0])
    w = np.array([4, 3])      # Original weight
    v = np.array([1, 0.5])    # Refusal direction (unnormalized for visual)
    v = v / np.linalg.norm(v) * 3.5 # Normalize and scale
    
    # Projection of w onto v
    proj_len = np.dot(w, v / np.linalg.norm(v))
    proj = (v / np.linalg.norm(v)) * proj_len
    
    # Ablated weight w' = w - proj
    w_prime = w - proj
    
    # Plot vectors
    ax.quiver(*origin, *w, color='#00ffcc', scale=1, scale_units='xy', angles='xy', width=0.015, label='Original Weight (W)')
    ax.quiver(*origin, *v, color='#ff0066', scale=1, scale_units='xy', angles='xy', width=0.015, label='Refusal Direction (v̂)')
    ax.quiver(*origin, *proj, color='#ffff00', scale=1, scale_units='xy', angles='xy', width=0.012, label='Projection (W · v̂)v̂ᵀ', alpha=0.7)
    ax.quiver(*origin, *w_prime, color='#00ccff', scale=1, scale_units='xy', angles='xy', width=0.015, label="Ablated Weight (W')")
    
    # Draw projection line (solid)
    ax.plot([w[0], proj[0]], [w[1], proj[1]], 'w-', alpha=0.5)
    
    # Annotations
    ax.text(w[0]+0.1, w[1]+0.1, 'W', color='#00ffcc', fontsize=14, fontweight='bold')
    ax.text(v[0]+0.1, v[1]-0.2, 'v̂', color='#ff0066', fontsize=14, fontweight='bold')
    ax.text(w_prime[0]-0.3, w_prime[1]+0.2, "W' = W - α(W·v̂)v̂ᵀ", color='#00ccff', fontsize=14, fontweight='bold')
    
    # Style
    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 5)
    ax.grid(False) # Disable grid to avoid dash errors
    ax.set_aspect('equal')
    ax.set_title('Directional Ablation Mechanism', fontsize=16, pad=20, color='white')
    ax.legend(loc='upper left', fontsize=10, facecolor='#1a1a1a', edgecolor='white')
    ax.axis('off')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#121212')
    plt.close()

def create_architecture_diagram(output_path):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    
    # Colors
    box_color = '#1e1e1e'
    edge_color = '#00ffcc'
    text_color = 'white'
    
    # Helper to draw box
    def draw_box(x, y, width, height, label, sublabel=""):
        rect = patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.1", 
                                      linewidth=2, edgecolor=edge_color, facecolor=box_color)
        ax.add_patch(rect)
        ax.text(x + width/2, y + height/2 + 0.15, label, ha='center', va='center', color=text_color, fontsize=10, fontweight='bold')
        if sublabel:
            ax.text(x + width/2, y + height/2 - 0.25, sublabel, ha='center', va='center', color='#aaaaaa', fontsize=8)
        return (x + width, y + height/2), (x, y + height/2) # Right, Left anchor

    # Nodes
    # Left column
    _, l_cli = draw_box(0.5, 4.5, 2.5, 1.0, "CLI Interface", "unfetter ablate")
    
    # Middle column (Backends)
    r_cpu, l_cpu = draw_box(4.0, 4.5, 2.0, 1.0, "Backend", "CPU/GPU/Distrib")
    
    # Right column (Process)
    r_load, l_load = draw_box(7.0, 4.5, 2.5, 1.0, "Load Model", "Quantized / Full")
    r_vec, l_vec = draw_box(7.0, 3.0, 2.5, 1.0, "Refusal Vector", "Difference-of-Means")
    r_ablate, l_ablate = draw_box(7.0, 1.5, 2.5, 1.0, "Ablation Core", "Layer-wise Projection")
    r_save, l_save = draw_box(7.0, 0.0, 2.5, 1.0, "Save Model", "SafeTensors")
    
    # Arrows
    def arrow(start, end):
        ax.annotate("", xy=end, xytext=start, 
                    arrowprops=dict(arrowstyle="->", color='white', lw=1.5, mutation_scale=15))

    arrow((3.1, 5.0), (3.9, 5.0)) # CLI -> Backend
    arrow((6.1, 5.0), (6.9, 5.0)) # Backend -> Load
    arrow((8.25, 4.4), (8.25, 4.1)) # Load -> Vector
    arrow((8.25, 2.9), (8.25, 2.6)) # Vector -> Ablate
    arrow((8.25, 1.4), (8.25, 1.1)) # Ablate -> Save

    # Title
    ax.text(6, 5.8, "Model Unfetter Architecture", ha='center', color='white', fontsize=18, fontweight='bold')
    
    ax.axis('off')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#121212')
    plt.close()

if __name__ == "__main__":
    output_dir = Path("assets")
    output_dir.mkdir(exist_ok=True)
    
    setup_style()
    print("Generating vector diagram...")
    create_vector_diagram(output_dir / "vector_projection.png")
    print("Generating architecture diagram...")
    create_architecture_diagram(output_dir / "architecture.png")
    print("Done.")
