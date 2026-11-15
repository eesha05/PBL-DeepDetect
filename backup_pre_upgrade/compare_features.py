import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Load extracted edge features
# --------------------------------------------------

csv_path = "outputs/deepdetect_edge_features.csv"

data = pd.read_csv(csv_path)

print("Feature dataset loaded successfully.")
print(data.head())


# --------------------------------------------------
# 2. Separate clean and stego images
# --------------------------------------------------

clean_data = data[data["image_type"] == "clean"]
stego_data = data[data["image_type"] == "stego"]


# --------------------------------------------------
# 3. Select edge-based features
# --------------------------------------------------

edge_features = [
    "sobel_mean",
    "sobel_std",
    "laplacian_mean",
    "laplacian_variance",
    "canny_edge_density"
]


# --------------------------------------------------
# 4. Calculate average edge features
# --------------------------------------------------

clean_means = clean_data[edge_features].mean()
stego_means = stego_data[edge_features].mean()

comparison_table = pd.DataFrame({
    "Clean Images": clean_means,
    "PVD Stego Images": stego_means
})

print("\nEdge-based feature comparison:")
print(comparison_table)


# --------------------------------------------------
# 5. Save the comparison table
# --------------------------------------------------

comparison_table.to_csv(
    "outputs/edge_based_comparison.csv"
)

print(
    "\nSaved table at: "
    "outputs/edge_based_comparison.csv"
)


# --------------------------------------------------
# 6. Plot edge-based comparison
# --------------------------------------------------

ax = comparison_table.plot(
    kind="bar",
    figsize=(12, 7)
)

ax.set_title("Edge-Based Comparison of Clean and PVD-Stego Images")
ax.set_xlabel("Edge-Based Features")
ax.set_ylabel("Average Feature Value")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    "outputs/edge_based_comparison.png",
    dpi=300
)

plt.show()

print(
    "Saved graph at: "
    "outputs/edge_based_comparison.png"
)