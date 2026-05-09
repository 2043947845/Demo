import os
import subprocess
import glob
import re

model_paths = glob.glob(r'runs\train\*\weights\best.pt')

print("Found {} models.".format(len(model_paths)))

results = []

for model_path in model_paths:
    print(f"Evaluating {model_path} ...")
    # Reduced warmup and testtime for CPU execution speed, set device to cpu
    cmd = f'python script/getFPS.py --weights {model_path} --device cpu --warmup 5 --testtime 20 --batch 1'
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    try:
        process = subprocess.Popen(cmd, shell=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        
        # Parse the output line, e.g., "model weights:runs\train\r50\weights\best.pt size:41.3M (bs:4)Latency:0.90165s +- 0.07073s fps:1.1"
        match = re.search(r'size:(.*?M) \(bs:(\d+)\)Latency:(.*?)s \+- (.*?)s fps:(.*?)$', stdout, re.MULTILINE)
        if match:
            size, bs, latency, std_time, fps = match.groups()
            results.append({
                'Model Path': model_path,
                'Size': size,
                'Batch Size': bs,
                'Latency (s/img)': latency,
                'Std (s)': std_time,
                'FPS': fps
            })
            print(f"Success: {model_path} - Latency: {latency}s, FPS: {fps}")
        else:
            print(f"Failed to parse output for {model_path}:\n{stdout}\n{stderr}")
    except Exception as e:
        print(f"Exception on {model_path}: {e}")

# Write results to markdown
with open('fps_results.md', 'w', encoding='utf-8') as f:
    f.write("# Model Inference Speed and FPS Results\n\n")
    f.write("Tested on CPU (`--device cpu`) with `batch=1`, `warmup=5`, `testtime=20` to reduce execution time.\n\n")
    f.write("| Model Path | Weight Size | Batch Size | Latency (s/img) | Std Deviation (s) | FPS |\n")
    f.write("| --- | --- | --- | --- | --- | --- |\n")
    for r in results:
        f.write(f"| `{r['Model Path']}` | {r['Size']} | {r['Batch Size']} | {r['Latency (s/img)']} | {r['Std (s)']} | **{r['FPS']}** |\n")

print("Evaluation complete. Written to fps_results.md.")
