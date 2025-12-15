import os
import pandas as pd

# -----------------------------
# Paths
# -----------------------------
sweep_csv = "CNN/sweep_labels_approx.csv"
discrete_root = os.getcwd() 
output_csv = "CNN/all_labels.csv"

# -----------------------------
# Load sweep CSV
# -----------------------------
df_sweep = pd.read_csv(sweep_csv)
df_sweep["source"] = "sweep"

# -----------------------------
# Prepare list for discrete images
# -----------------------------
rows = []

# -----------------------------
# Loop through discrete folders
# Folder names must be like:
#   1_hertz
#   2_hertz
#   4_hertz
#   8_hertz
#   16_hertz
# -----------------------------
for folder in os.listdir(discrete_root):
    if "_hertz" not in folder:
        continue

    freq = int(folder.split("_")[0])
    folder_path = os.path.join(discrete_root, folder)

    for file in os.listdir(folder_path):
        if file.lower().endswith((".bmp")):
            rows.append([file, freq, "discrete"])

# Create dataframe for discrete images
# -----------------------------
df_discrete = pd.DataFrame(rows, columns=["filename", "frequency_hz", "source"])

df_all = pd.concat([df_sweep, df_discrete], ignore_index=True)
df_all.to_csv("all_labels.csv", index=False)


# -----------------------------
# Combine sweep + discrete
# -----------------------------
df_all = pd.concat([df_sweep, df_discrete], ignore_index=True)

# -----------------------------
# Save final CSV
# -----------------------------
df_all.to_csv(output_csv, index=False)

print("Done! Combined CSV saved as:", output_csv)
print("Sweep images:", len(df_sweep))
print("Discrete images:", len(df_discrete))
print("Total images:", len(df_all))