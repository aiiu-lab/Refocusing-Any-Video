import torchmetrics
import os
import torch
from PIL import Image
import numpy as np
import csv
import sys

num_positions = 9

output_dir_path = "./results/focal_stacks"



gt = "gt"
model = sys.argv[1]

gt_path = os.path.join(output_dir_path, gt)
model_path = os.path.join(output_dir_path, model)

device = sys.argv[2]

metrics_grid = []

for i in range(num_positions):
    row = []
    for j in range(num_positions):
        metrics = {
            "psnr": torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0).to(device),
            "ssim": torchmetrics.image.StructuralSimilarityIndexMeasure().to(device),
            "lpips": torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(device),
            "fid": torchmetrics.image.fid.FrechetInceptionDistance(normalize=True).to(device),
            "vif": torchmetrics.image.VisualInformationFidelity().to(device),
        }
        row.append(metrics)
    metrics_grid.append(row)
    print("Created metrics for position", i)

#lopp through each directory in gt_path
#get all directories in gt_path
position_dirs = os.listdir(gt_path)
position_dirs = sorted([dir for dir in position_dirs if os.path.isdir(os.path.join(gt_path, dir))]) [0:num_positions]

for gt_dir in position_dirs:
    position_number = int(gt_dir.split("_")[1])
    #get pngs inside that directory
    gt_pngs = sorted(os.listdir(os.path.join(gt_path, gt_dir, "images")))
    #Confirm that number of pngs == 164*9
    assert len(gt_pngs) == 164*9
    #loop through the 164 imgs
    for i in range(164):
        #get the 9 frames
        gt_frames_names = gt_pngs[i*9:(i+1)*9]
        #load the 9 frames
        gt_frames = [Image.open(os.path.join(gt_path, gt_dir, "images", frame)) for frame in gt_frames_names]
        #make into numpy arraymo
        gt_frames = [torch.tensor(np.array(frame)/255).to(torch.float32).to(device).permute(2,0,1).unsqueeze(0) for frame in gt_frames]
        
        #load model_frames which is almost smae path but in model_path
        model_frames = [Image.open(os.path.join(model_path, gt_dir, "images", frame)) for frame in gt_frames_names]
        #make into numpy array
        model_frames = [torch.tensor(np.array(frame)/255).to(torch.float32).to(device).permute(2,0,1).unsqueeze(0) for frame in model_frames]

        #loop through the 9 frames
        for j in range(num_positions):
            #compute metrics
            for key, metric in metrics_grid[position_number][j].items():
                #if frames have a 4th channel discard it
                if gt_frames[j].shape[1] == 4:
                    gt_frames[j] = gt_frames[j][:,:3,:,:]
                if model_frames[j].shape[1] == 4:
                    model_frames[j] = model_frames[j][:,:3,:,:]
                if key == "fid":
                    metric.update(model_frames[j], real=False)
                    metric.update(gt_frames[j], real=True)
                else:
                    metric(gt_frames[j], model_frames[j])

        print("Computed metrics for position", position_number, "frame", i)

#write the metrics to a csv (each metric as a csv)

def write_metrics_to_csv(metrics_grid, metric_names, formatting_options=None, output_dir="metrics_output"):
    """
    Writes each metric in the metrics_grid to a separate CSV file.

    Args:
        metrics_grid (list): A 9x9 list of dictionaries containing metrics.
        metric_names (list): List of metric names (e.g., ["psnr", "lpips", "fid"]).
        output_dir (str): Directory where the CSV files will be saved.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

    positions = list(range(1, num_positions+1))

    for metric_name in metric_names:
        output_file = os.path.join(output_dir, f"{metric_name}.csv")

        # Get the formatting function for the current metric, or use default
        format_fn = formatting_options.get(metric_name, lambda x: f"{x}") if formatting_options else lambda x: f"{x}"
        
        
        # Write the metric to the CSV
        with open(output_file, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)

            header = ["Starting Position/End Position"] + [f"Position {i}" for i in positions]
            writer.writerow(header)
            
            # Iterate over the grid and extract the metric values
            for i, row in enumerate(metrics_grid):
                csv_row = [f"Position {positions[i]}"]  # Add the column label as the first column
                for cell in row:
                    metric = cell[metric_name]
                    # Assuming metrics are PyTorch objects with a `compute` method
                    # Replace `0.0` with metric.compute() if metric values are computed
                    value = 0.0 if not hasattr(metric, "compute") else metric.compute().item()
                    csv_row.append(format_fn(value))  # Format the value
                writer.writerow(csv_row)
                print(f"Wrote row for position {positions[i]} with metric {metric_name}")
        
        print(f"Saved {metric_name} metrics to {output_file}")

formatting_options = {
    "psnr": lambda x: f"{x:.2f}",  # Two decimal places
    "lpips": lambda x: f"{x:.4f}",  # Four decimal places
    "fid": lambda x: f"{x:.2f}",  # Two  decimal places
    "ssim": lambda x: f"{x:.4f}",  # Four decimal places
    "vif": lambda x: f"{x:.4f}"  # Four decimal places
}



write_metrics_to_csv(metrics_grid, ["psnr", "ssim", "lpips", "fid", "vif"], formatting_options=formatting_options, output_dir=f"{output_dir_path}/metrics_output/{model}")
        
