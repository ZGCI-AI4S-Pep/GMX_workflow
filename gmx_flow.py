#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import shutil
import subprocess
import time
import multiprocessing
import queue # Used for GPU ID queue management
import re # Used for parsing nvidia-smi output

# --- Configuration ---
PRE_FOLDER = "Pre"  # Name of the folder containing input files
MAX_CONCURRENT_PROCESSES = 4  # Number of simulations to run in parallel (corresponds to parameter 'a')
GPU_CHECK_INTERVAL = 10 # Seconds to wait before checking for available GPUs or GPU status again if queue is empty
GPU_BUSY_CHECK_INTERVAL = 10 # Seconds to wait if a GPU is found but is busy (utilization >= threshold)
GPU_NON_ZERO_WAIT = 1 # Seconds to wait if a GPU is found with >0% utilization but < threshold before trying again (soft prioritization)
# List of GPU IDs to use. Modify this list if you want to use specific GPUs,
# e.g., [0, 1, 3]. If empty, the script will attempt to detect available GPUs.
# Manually provide if detection fails or you want to use all GPUs.
AVAILABLE_GPU_IDS = []
# GPU utilization threshold (percentage). A GPU is considered busy if its utilization is above this value.
GPU_UTILIZATION_THRESHOLD = 60 # Percentage

# --- Helper Functions ---

def run_command(cmd_list, cwd, log_prefix, input_str=None):
    """Runs a shell command using subprocess and logs output."""
    cmd_str = ' '.join(cmd_list)
    print(f"[{log_prefix}] Running command in {cwd}: {cmd_str}", flush=True)
    try:
        process = subprocess.run(
            cmd_list,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            input=input_str,
            encoding='utf-8' # Attempt to specify encoding
        )
        # print(f"[{log_prefix}] STDOUT:\n{process.stdout}", flush=True) # Standard output can be verbose, commented by default
        if process.stderr:
            print(f"[{log_prefix}] STDERR:\n{process.stderr}", flush=True) # Print standard error
        print(f"[{log_prefix}] Command successful: {cmd_str}", flush=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[{log_prefix}] Error running command: {cmd_str}", flush=True)
        print(f"[{log_prefix}] Return code: {e.returncode}", flush=True)
        print(f"[{log_prefix}] STDOUT:\n{e.stdout}", flush=True)
        print(f"[{log_prefix}] STDERR:\n{e.stderr}", flush=True)
        return False
    except FileNotFoundError:
        print(f"[{log_prefix}] Error: Command not found: {cmd_list[0]}. Please ensure GROMACS, obabel, acpype, and python are in your PATH.", flush=True)
        return False
    except Exception as e:
        print(f"[{log_prefix}] Unexpected error running {cmd_str}: {e}", flush=True)
        return False

def get_gpu_ids_from_nvidia_smi():
    """Attempts to get GPU indices from nvidia-smi."""
    gpu_ids = []
    try:
        # Try running nvidia-smi with UTF-8 encoding
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
            capture_output=True, text=True, check=True, encoding='utf-8'
        )
        output = result.stdout.strip()
        for line in output.split('\n'):
            if line:
                gpu_ids.append(int(line.strip()))
        print(f"Detected GPU IDs using nvidia-smi: {gpu_ids}")
    except FileNotFoundError:
        print("Warning: 'nvidia-smi' command not found. Cannot auto-detect GPUs.")
    except subprocess.CalledProcessError as e:
        print(f"Warning: Error running nvidia-smi: {e}. Stderr: {e.stderr}")
    except Exception as e:
        print(f"Warning: Error parsing nvidia-smi output: {e}")
    return gpu_ids

def get_gpu_utilization(gpu_id):
    """Attempts to get the utilization of a specific GPU from nvidia-smi."""
    try:
        # Query utilization.gpu [%] for the specific GPU ID
        result = subprocess.run(
            ['nvidia-smi', f'--id={gpu_id}', '--query-gpu=utilization.gpu', '--format=csv,noheader'],
            capture_output=True, text=True, check=True, encoding='utf-8'
        )
        output = result.stdout.strip()
        # Expected output format: "XX %"
        match = re.match(r"(\d+)\s*%", output)
        if match:
            utilization = int(match.group(1))
            return utilization
        else:
            print(f"Warning: Could not parse nvidia-smi utilization output for GPU {gpu_id}: {output}", flush=True)
            return -1 # Indicate error
    except FileNotFoundError:
        print("Warning: 'nvidia-smi' command not found. Cannot check GPU utilization.", flush=True)
        return -1 # Indicate error
    except subprocess.CalledProcessError as e:
        print(f"Warning: Error running nvidia-smi for GPU {gpu_id} utilization: {e}. Stderr: {e.stderr}", flush=True)
        return -1 # Indicate error
    except Exception as e:
        print(f"Warning: Unexpected error checking GPU {gpu_id} utilization: {e}", flush=True)
        return -1 # Indicate error

def run_gpu_command(base_cmd_list, cwd, log_prefix, gpu_queue, utilization_threshold, non_zero_wait, busy_wait, check_interval):
    """Acquires a GPU ID, checks its utilization, runs the command if idle, and releases the ID."""
    gpu_id = None
    while True: # Keep trying until a GPU is acquired and used
        try:
            # Attempt to get a GPU ID from the queue
            # Block until an ID is available, with a timeout
            gpu_id = gpu_queue.get(block=True, timeout=check_interval)
            print(f"[{log_prefix}] Acquired potential GPU {gpu_id} from queue.", flush=True)

            # Check GPU utilization
            utilization = get_gpu_utilization(gpu_id)

            if utilization == -1:
                # Error checking utilization, release GPU and try again
                print(f"[{log_prefix}] Error checking utilization for GPU {gpu_id}. Releasing GPU and trying again.", flush=True)
                gpu_queue.put(gpu_id) # Put it back
                gpu_id = None # Reset to None to loop and get another ID
                time.sleep(busy_wait) # Wait before trying again
                continue # Go back to the start of the while loop

            if utilization == 0:
                # GPU is idle (0% utilization), prioritize using it
                print(f"[{log_prefix}] GPU {gpu_id} is idle (0% utilization). Proceeding.", flush=True)
                # Build the full command including the acquired GPU ID
                cmd_list = base_cmd_list + ["-gpu_id", str(gpu_id)]

                # Run the command
                success = run_command(cmd_list, cwd=cwd, log_prefix=log_prefix)

                # Command finished, put the GPU ID back
                gpu_queue.put(gpu_id)
                print(f"[{log_prefix}] Released GPU {gpu_id}", flush=True)
                return success # Return command success status

            elif utilization < utilization_threshold:
                # GPU is not idle but below the busy threshold.
                # Put it back and wait a short time before trying again,
                # allowing a 0% GPU to potentially be picked up by this or another process.
                print(f"[{log_prefix}] GPU {gpu_id} is available ({utilization}% utilization, below threshold). Releasing GPU and waiting briefly...", flush=True)
                gpu_queue.put(gpu_id) # Put the ID back into the queue
                gpu_id = None # Reset to None
                time.sleep(non_zero_wait) # Wait a short time
                continue # Go back to the start of the while loop

            else: # utilization >= utilization_threshold
                # GPU is busy, put it back and wait the busy interval
                print(f"[{log_prefix}] GPU {gpu_id} is busy ({utilization}% utilization). Releasing GPU and waiting...", flush=True)
                gpu_queue.put(gpu_id) # Put the ID back into the queue
                gpu_id = None # Reset to None
                time.sleep(busy_wait) # Wait before trying to get another ID
                continue # Go back to the start of the while loop

        except queue.Empty:
            # No available GPU ID in the queue within the timeout
            print(f"[{log_prefix}] No available GPU IDs in queue. Waiting {check_interval} seconds...", flush=True)
            gpu_id = None # Ensure gpu_id remains None to loop again
            # The timeout in get() handles the waiting interval

        except Exception as e:
            print(f"[{log_prefix}] Unexpected error during GPU command execution: {e}", flush=True)
            # If an error occurs *after* acquiring the GPU ID, we should release it
            if gpu_id is not None:
                print(f"[{log_prefix}] Releasing GPU {gpu_id} due to unexpected error.", flush=True)
                gpu_queue.put(gpu_id) # Put it back into the queue
            return False # Indicate failure

        # If we reach here without returning, it means queue.Empty occurred or an error happened before acquiring gpu_id.
        # The loop continues to try again.


def run_simulation_workflow(mol_filepath, pre_dir_abs, base_work_dir, gpu_queue, utilization_threshold, non_zero_wait, busy_wait, check_interval):
    """Runs the entire simulation workflow for a single .mol file."""
    base_name = os.path.splitext(os.path.basename(mol_filepath))[0]
    work_dir = os.path.join(base_work_dir, base_name)
    log_prefix = base_name # Use the base name as the log prefix for clarity

    print(f"[{log_prefix}] Starting processing file {mol_filepath}", flush=True)

    # --- Check if workflow is already completed ---
    md_log_path = os.path.join(work_dir, "md.log")
    md_xtc_path = os.path.join(work_dir, "md.xtc")
    if os.path.exists(md_log_path) and os.path.exists(md_xtc_path):
        print(f"[{log_prefix}] Workflow appears completed (md.log and md.xtc found). Skipping.", flush=True)
        return # Skip this molecule if already processed

    print(f"[{log_prefix}] Workflow not completed, proceeding with processing.", flush=True)

    try:
        # --- 1. Setup Directory ---
        os.makedirs(work_dir, exist_ok=True)
        print(f"[{log_prefix}] Created work directory: {work_dir}", flush=True)

        # --- 2. Copy files from Pre folder ---
        if not os.path.isdir(pre_dir_abs):
            print(f"[{log_prefix}] Error: Pre directory '{pre_dir_abs}' not found.", flush=True)
            return # Stop processing this molecule
        try:
            for item in os.listdir(pre_dir_abs):
                s = os.path.join(pre_dir_abs, item)
                d = os.path.join(work_dir, item)
                if os.path.isfile(s):
                    shutil.copy2(s, d)
            # Also copy the original .mol file itself
            shutil.copy2(mol_filepath, os.path.join(work_dir, os.path.basename(mol_filepath)))
            print(f"[{log_prefix}] Copied files from {pre_dir_abs} to {work_dir}", flush=True)
            # Rename this workflow-specific mol file to exam.mol
            original_mol_in_workdir = os.path.join(work_dir, os.path.basename(mol_filepath))
            exam_mol_path = os.path.join(work_dir, "exam.mol")
            if os.path.exists(original_mol_in_workdir):
                os.rename(original_mol_in_workdir, exam_mol_path)
                print(f"[{log_prefix}] Renamed {os.path.basename(mol_filepath)} to exam.mol", flush=True)
            else:
                print(f"[{log_prefix}] Error: Could not find {os.path.basename(mol_filepath)} in {work_dir} after copy.", flush=True)
                return # Stop if rename fails

        except Exception as e:
            print(f"[{log_prefix}] Error copying files: {e}", flush=True)
            return # Stop processing this molecule

        # --- 3. Run Simulation Steps ---
        # Note: Assumes gmx_mpi, obabel, acpype etc. commands are in PATH
        # Note: Assumes tip4p.gro is accessible (in PATH, GMXDATA, or copied from Pre)

        # Step 1: obabel (Generate exam.mol2)
        if not run_command(["obabel", "exam.mol", "-O", "exam.mol2", "-h"], cwd=work_dir, log_prefix=log_prefix): return

        # Step 2: python correctmol2.py (Correct exam.mol2)
        if not run_command(["python", "correctmol2.py"], cwd=work_dir, log_prefix=log_prefix): return

        # Step 3: acpype (Generate topology)
        if not run_command(["acpype", "-i", "exam.mol2", "-c", "gas", "-n", "0", "-b", "exam", "-o", "gmx", "-d", "-f"], cwd=work_dir, log_prefix=log_prefix): return

        # Step 3b: Copy acpype results
        try:
            acpype_tmp_pdb = os.path.join(work_dir, ".acpype_tmp_exam", "tmp")
            target_pdb = os.path.join(work_dir, "exam.pdb")
            acpype_itp = os.path.join(work_dir, "exam.acpype", "exam_GMX.itp")
            target_itp = os.path.join(work_dir, "exam_GMX.itp")
            acpype_posre = os.path.join(work_dir, "exam.acpype", "posre_exam.itp")
            target_posre = os.path.join(work_dir, "posre_exam.itp")

            if os.path.exists(acpype_tmp_pdb):
                shutil.copy2(acpype_tmp_pdb, target_pdb)
                print(f"[{log_prefix}] Copied {acpype_tmp_pdb} to {target_pdb}", flush=True)
            else:
                print(f"[{log_prefix}] Error: Could not find {acpype_tmp_pdb}", flush=True)
                return

            if os.path.exists(acpype_itp):
                shutil.copy2(acpype_itp, target_itp)
                print(f"[{log_prefix}] Copied {acpype_itp} to {target_itp}", flush=True)
            else:
                print(f"[{log_prefix}] Error: Could not find {acpype_itp}", flush=True)
                return

            if os.path.exists(acpype_posre):
                shutil.copy2(acpype_posre, target_posre)
                print(f"[{log_prefix}] Copied {acpype_posre} to {target_posre}", flush=True)
            else:
                print(f"[{log_prefix}] Error: Could not find {acpype_posre}", flush=True)
                return
        except Exception as e:
            print(f"[{log_prefix}] Error copying acpype results: {e}", flush=True)
            return

        # Step 4: gmx editconf (Create box)
        if not run_command(["gmx_mpi", "editconf", "-f", "exam.pdb", "-o", "exam.gro", "-c", "-d", "1.0", "-bt", "cubic"], cwd=work_dir, log_prefix=log_prefix): return

        # Step 5: python correctgro.py (Correct exam.gro)
        if not run_command(["python", "correctgro.py"], cwd=work_dir, log_prefix=log_prefix): return

        # Step 6: gmx solvate (Add solvent)
        # Ensure tip4p.gro is available. If it's not a standard file, add it to the 'Pre' folder.
        if not run_command(["gmx_mpi", "solvate", "-cp", "exam.gro", "-cs", "tip4p.gro", "-o", "exam_sol.gro", "-p", "exam_GMX.top"], cwd=work_dir, log_prefix=log_prefix): return

        # Step 7: gmx grompp (Prepare for adding ions)
        if not run_command(["gmx_mpi", "grompp", "-f", "em.mdp", "-c", "exam_sol.gro", "-p", "exam_GMX.top", "-o", "ion.tpr", "-maxwarn", "2"], cwd=work_dir, log_prefix=log_prefix): return # Added maxwarn based on step 7

        # Step 8: gmx genion (Add ions)
        if not run_command(["gmx_mpi", "genion", "-s", "ion.tpr", "-o", "exam_sol_ions.gro", "-p", "exam_GMX.top", "-conc", "0.15", "-neutral", "-pname", "NA", "-nname", "CL"], cwd=work_dir, log_prefix=log_prefix, input_str="SOL\n"): return # Added newline to input_str

        # Step 9: gmx grompp (Prepare for Energy Minimization EM)
        if not run_command(["gmx_mpi", "grompp", "-f", "em.mdp", "-c", "exam_sol_ions.gro", "-p", "exam_GMX.top", "-o", "em.tpr", "-maxwarn", "2"], cwd=work_dir, log_prefix=log_prefix): return # Added maxwarn

        # --- GPU Steps ---
        # Step 10: gmx mdrun (Energy Minimization) - Requires GPU
        if not run_gpu_command(["gmx_mpi", "mdrun", "-deffnm", "em"], cwd=work_dir, log_prefix=log_prefix, gpu_queue=gpu_queue, utilization_threshold=utilization_threshold, non_zero_wait=non_zero_wait, busy_wait=busy_wait, check_interval=check_interval): return

        # Step 11: gmx grompp (Prepare for NVT equilibration)
        if not run_command(["gmx_mpi", "grompp", "-f", "nvt.mdp", "-c", "em.gro", "-r", "em.gro", "-p", "exam_GMX.top", "-o", "nvt.tpr", "-maxwarn", "2"], cwd=work_dir, log_prefix=log_prefix): return # Added maxwarn

        # Step 12: gmx mdrun (NVT equilibration) - Requires GPU
        # Note: Added specific GPU flags based on common GROMACS usage
        if not run_gpu_command(["gmx_mpi", "mdrun", "-deffnm", "nvt", "-ntomp", "12", "-nb", "gpu", "-pme", "gpu", "-bonded", "gpu", "-pin", "on"], cwd=work_dir, log_prefix=log_prefix, gpu_queue=gpu_queue, utilization_threshold=utilization_threshold, non_zero_wait=non_zero_wait, busy_wait=busy_wait, check_interval=check_interval): return

        # Step 13: gmx grompp (Prepare for NPT equilibration)
        if not run_command(["gmx_mpi", "grompp", "-f", "npt.mdp", "-c", "nvt.gro", "-r", "nvt.gro", "-p", "exam_GMX.top", "-o", "npt.tpr", "-maxwarn", "2"], cwd=work_dir, log_prefix=log_prefix): return # Added maxwarn

        # Step 14: gmx mdrun (NPT equilibration) - Requires GPU
        # Note: Added specific GPU flags based on common GROMACS usage
        if not run_gpu_command(["gmx_mpi", "mdrun", "-deffnm", "npt", "-ntomp", "12", "-nb", "gpu", "-pme", "gpu", "-bonded", "gpu", "-pin", "on"], cwd=work_dir, log_prefix=log_prefix, gpu_queue=gpu_queue, utilization_threshold=utilization_threshold, non_zero_wait=non_zero_wait, busy_wait=busy_wait, check_interval=check_interval): return

        # Step 15: gmx grompp (Prepare for Production MD)
        if not run_command(["gmx_mpi", "grompp", "-f", "md.mdp", "-c", "npt.gro", "-r", "npt.gro", "-p", "exam_GMX.top", "-o", "md.tpr", "-maxwarn", "2"], cwd=work_dir, log_prefix=log_prefix): return # Added maxwarn

        # Step 16: gmx mdrun (Production MD) - Requires GPU
        # Note: Added specific GPU flags based on common GROMACS usage
        if not run_gpu_command(["gmx_mpi", "mdrun", "-deffnm", "md", "-ntomp", "12", "-nb", "gpu", "-pme", "gpu", "-bonded", "gpu", "-pin", "on"], cwd=work_dir, log_prefix=log_prefix, gpu_queue=gpu_queue, utilization_threshold=utilization_threshold, non_zero_wait=non_zero_wait, busy_wait=busy_wait, check_interval=check_interval): return

        print(f"[{log_prefix}] Workflow completed successfully for {base_name}", flush=True)

    except Exception as e:
        print(f"[{log_prefix}] Unexpected error processing workflow for {base_name}: {e}", flush=True)
        # Add cleanup or more detailed logging here if needed
    finally:
        # Optional: Add cleanup steps here if needed
        pass

# --- Main Execution ---
if __name__ == "__main__":
    # Get the absolute path of the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Base directory where .mol files are located and where subdirectories will be created
    base_work_dir = os.getcwd()
    pre_dir_abs = os.path.join(base_work_dir, PRE_FOLDER)

    print(f"Starting GROMACS workflow automation script.")
    print(f"Base directory: {base_work_dir}")
    print(f"Pre folder: {pre_dir_abs}")
    print(f"Maximum concurrent processes: {MAX_CONCURRENT_PROCESSES}")
    print(f"GPU utilization threshold for 'busy': {GPU_UTILIZATION_THRESHOLD}%")
    print(f"Interval to check GPU status: {GPU_CHECK_INTERVAL} seconds")
    print(f"Interval to wait if GPU is busy: {GPU_BUSY_CHECK_INTERVAL} seconds")
    print(f"Interval to wait if GPU is >0% but <threshold: {GPU_NON_ZERO_WAIT} seconds (for 0% prioritization)")


    # --- Find .mol files ---
    mol_files = glob.glob(os.path.join(base_work_dir, "*.mol"))
    if not mol_files:
        print("Error: No *.mol files found in the current directory.")
        exit(1)
    print(f"Found {len(mol_files)} .mol files to process:")
    for f in mol_files:
        print(f"  - {os.path.basename(f)}")

    # --- Determine GPU IDs ---
    if not AVAILABLE_GPU_IDS:
        print("Attempting to detect GPU IDs using nvidia-smi...")
        gpu_ids_to_use = get_gpu_ids_from_nvidia_smi()
    else:
        gpu_ids_to_use = AVAILABLE_GPU_IDS
        print(f"Using predefined GPU IDs: {gpu_ids_to_use}")

    if not gpu_ids_to_use:
        print("Error: No GPU IDs specified or detected. GPU steps cannot run.")
        # Decide whether to exit or continue without GPU steps
        # For this script, GPU steps are essential, so we exit.
        exit(1)
    print(f"Will use GPU IDs: {gpu_ids_to_use}")


    # --- Setup Multiprocessing ---
    # Use Manager to create a shared GPU ID queue
    manager = multiprocessing.Manager()
    gpu_queue = manager.Queue()

    # Populate the queue with available GPU IDs
    for gpu_id in gpu_ids_to_use:
        gpu_queue.put(gpu_id)

    # Prepare arguments for each workflow task
    tasks = []
    for mol_file in mol_files:
        # Pass all relevant configuration parameters to the worker function
        tasks.append((mol_file, pre_dir_abs, base_work_dir, gpu_queue,
                      GPU_UTILIZATION_THRESHOLD, GPU_NON_ZERO_WAIT,
                      GPU_BUSY_CHECK_INTERVAL, GPU_CHECK_INTERVAL))

    # --- Run Workflows in Parallel ---
    print(f"\nStarting processing pool with {MAX_CONCURRENT_PROCESSES} worker processes...")
    try:
        # Use a process pool to limit the number of concurrently running tasks
        with multiprocessing.Pool(processes=MAX_CONCURRENT_PROCESSES) as pool:
            # Use starmap to pass multiple arguments to the worker function
            pool.starmap(run_simulation_workflow, tasks)
    except Exception as e:
        print(f"Error occurred in the multiprocessing pool: {e}")
    finally:
        print("Waiting for all processes to complete...")
        # The Pool context manager automatically handles join()
        print("All processes have completed.")

    print("\nScript execution finished.")
