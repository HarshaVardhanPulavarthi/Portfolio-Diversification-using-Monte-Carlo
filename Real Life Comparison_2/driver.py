import subprocess
import re
import numpy as np

scripts = [f"compare_{i}.py" for i in range(1, 5)]
# Number of runs per script
runs_per_script = 25

# For each script
for script in scripts:
    avg_errors = []
    max_errors = []
    
    print(f"\nRunning {script.split('/')[-1]} {runs_per_script} times...")

    for run in range(runs_per_script):
        try:
            result = subprocess.run(["python", script], capture_output=True, text=True)
            output = result.stdout
            
            # Use regex to extract percentage errors from stdout
            avg_match = re.search(r"Average Percentage Error:\s*([\d\.]+)", output)
            max_match = re.search(r"Maximum Percentage Error:\s*([\d\.]+)", output)
            
            if avg_match and max_match:
                avg_error = float(avg_match.group(1))
                max_error = float(max_match.group(1))
                avg_errors.append(avg_error)
                max_errors.append(max_error)
            else:
                print(f"Run {run+1} of {script.split('/')[-1]} failed to parse output.")
                
        except Exception as e:
            print(f"Run {run+1} of {script.split('/')[-1]} failed: {e}")

    # Compute mean and std dev
    avg_errors = np.array(avg_errors)
    max_errors = np.array(max_errors)

    if len(avg_errors) > 0:
        print(f"\nSummary for {script.split('/')[-1]}:")
        print(f"  Mean Average Error: {avg_errors.mean():.2f}% ± {avg_errors.std():.2f}%")
        print(f"  Mean Maximum Error: {max_errors.mean():.2f}% ± {max_errors.std():.2f}%")
    else:
        print(f"No valid results for {script.split('/')[-1]}.")
