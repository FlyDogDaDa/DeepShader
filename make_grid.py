import glob
import os

from PIL import Image

samples_dir = "/home/b11223209/workspace/ProgramDevelopment/DeepShader/runs/transformer-4-subset-32_20260606_001759/samples"
output_path = "/home/b11223209/workspace/ProgramDevelopment/DeepShader/runs/transformer-4-subset-32_20260606_001759/samples_grid.png"

# Collect images per sample: sample_XXXX_idx[0-4].png
rows = []
for sample_idx in range(26):
    row_images = []
    for idx in range(5):
        filename = f"sample_{sample_idx:04d}_idx{idx}.png"
        path = os.path.join(samples_dir, filename)
        img = Image.open(path)
        row_images.append(img)
    # Horizontally stack the 5 images for this sample
    widths = [img.width for img in row_images]
    heights = [img.height for img in row_images]
    max_width = max(widths)
    max_height = max(heights)
    # Resize all to same dimensions first
    row_images = [img.resize((max_width, max_height)) for img in row_images]
    total_width = sum(widths)
    combined = Image.new("RGB", (total_width, max_height))
    x_offset = 0
    for img in row_images:
        combined.paste(img, (x_offset, 0))
        x_offset += img.width
    rows.append(combined)

# Determine grid dimensions
row_widths = [r.width for r in rows]
row_heights = [r.height for r in rows]
max_row_width = max(row_widths)
total_height = sum(row_heights)

# Vertally stack all rows
grid = Image.new("RGB", (max_row_width, total_height))
y_offset = 0
for row in rows:
    grid.paste(row, (0, y_offset))
    y_offset += row.height

grid.save(output_path, "PNG")
print(f"Grid saved to {output_path} ({max_row_width}x{total_height})")
