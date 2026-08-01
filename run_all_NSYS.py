import glob
import subprocess
import os
import sys

output_dir = "nsys_data"
os.makedirs(output_dir, exist_ok=True)

env = os.environ.copy()
env["PYTHONPATH"] = os.pathsep.join(sys.path)

files = [f for f in glob.glob("sim_*.py") if not os.path.basename(f).startswith("sim_cpu_")]
print("Scripts encontrados (GPU):", files)

for file in files:
    base_name = os.path.splitext(os.path.basename(file))[0]
    rep_file = os.path.join(output_dir, f"perfil_{base_name}.nsys-rep")

    print(f"\n### Profiling {file} com Nsight Systems ###")

    profile_cmd = [
        "nsys",
        "profile",
        "--trace=cuda,vulkan,osrt",
        "--gpu-metrics-devices=all",
        "--force-overwrite=true",
        f"--output={os.path.splitext(rep_file)[0]}",
        "python",
        file,
        "-c",
        os.path.join(".", "ensaios", "ponto", "ponto_sem_plots_1x.json")
    ]

    resultado = subprocess.run(profile_cmd, capture_output=True, text=True, env=env)
    print("STDOUT:\n", resultado.stdout)
    print("STDERR:\n", resultado.stderr)
