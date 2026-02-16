import time
from onnx_to_gurobi.onnxToGurobi import ONNXToGurobi
import gc
from utils import get_current_memory_mb, run_onnx_on_img
import config
from gurobipy import GRB
from data import get_mnist_batch
import numpy as np
import cv2
import pandas as pd
import tracemalloc

def ctf_evaluate():
    # ================= Setup ===================
    onnx_path = config.FC_MNIST_PATH
    iterations = config.ITERATIONS
    mnist_lb = config.MNIST_LB
    mnist_ub = config.MNIST_UB

    # ================= Setup GurobiModel
    input_node = "input"
    output_node = "output"
    ctf_output_path = "results/final/onnxgurobi/ctf_fc_mnist.csv"


    # ================= Load images and nn predictions
    images, labels = get_mnist_batch(iterations)
    ctf_targets = [0 for i in range(iterations)]
    results = []

    # ================ START ================

    tmp_constraints = []
    tmp_vars = []

    for i in range(iterations):
        gc.collect()

        tracemalloc.start()

        time_iteration_start = time.perf_counter()

        # ================= Setup GurobiModel
        model_builder = ONNXToGurobi(onnx_path)
        model_builder.build_model()
        gurobi_model = model_builder.get_gurobi_model()
        gurobi_model.setParam("OutputFlag", 0)
        gurobi_model_input_vars = model_builder.variables.get(input_node)
        gurobi_model_output_vars = model_builder.variables.get(output_node)
        gurobi_input_keys = sorted(gurobi_model_input_vars.keys())
        gurobi_output_keys = sorted(gurobi_model_output_vars.keys())
        gurobi_model.setAttr("LB", list(gurobi_model_input_vars.values()), 0.0)
        gurobi_model.setAttr("UB", list(gurobi_model_input_vars.values()), 1.0)

        # Set time limit and MIP focus of GurobiModel
        gurobi_model.params.MIPFocus = config.MIPFOCUS
        gurobi_model.params.TimeLimit = config.TIME_LIMIT

        img = images[i].clone().flatten()
        label = labels[i]
        target = ctf_targets[i]

        # Define dist vars
        dist_vars = gurobi_model.addVars(len(gurobi_model_input_vars), name="dist_vars")
        tmp_vars.append(dist_vars)
        for flat_idx, key_tuple in enumerate(gurobi_input_keys):
            gurobi_input_var = gurobi_model_input_vars[key_tuple]
            dist_var = dist_vars[flat_idx]
            # Realizing the L1 distance |x_var - x_original|
            c1 = gurobi_model.addConstr(dist_var >= img[flat_idx] - gurobi_input_var)
            c2 = gurobi_model.addConstr(dist_var >= gurobi_input_var - img[flat_idx])
            tmp_constraints.extend([c1, c2])

        # Define constraints ensuring class 0 is predicted
        target_var = gurobi_model_output_vars.get(gurobi_output_keys[target])
        for output_idx in range(10):
            if target != output_idx:
                gurobi_output_var = gurobi_model_output_vars.get(gurobi_output_keys[output_idx])
                c_target = gurobi_model.addConstr(target_var >= gurobi_output_var + 0.001)
                tmp_constraints.extend([c_target])

        # Set objective to minimize L1 distance btw. input and ctf
        gurobi_model.setObjective(dist_vars.sum(), GRB.MINIMIZE)

        # ================= MEASUREMENTS ======================

        # Start time and memory measurement for the optimization
        gurobi_model.optimize()

        # ================= Postprocessing ======================
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        mem_mb = current_mem / 10 ** 6

        time_iteration_end = time.perf_counter()
        time_duration = time_iteration_end - time_iteration_start

        print(f"Optimization of iteration {i} is finished!")

        try:
            gap = gurobi_model.MIPGap
        except AttributeError:
            gap = "Not accessible"

        results.append({
            "iteration": i,
            "gt_label": label,
            "target_label": target,
            "grb_Status": gurobi_model.status,
            "grb_Runtime": gurobi_model.runtime,
            "grb_ObjectiveValue": gurobi_model.objVal,
            "grb_MIPGap": gap,
            "grb_NumVars": gurobi_model.numVars,
            "grb_NumBinVars": gurobi_model.numBinVars,
            "iteration_time": time_duration,
            "iteration_mem_consumption": mem_mb
        })

    mem_eval_consumption = sum(result["iteration_mem_consumption"] for result in results)
    time_eval_duration = sum(result["iteration_time"] for result in results)
    results.append({
        "iteration": "-",
        "gt_label": "-",
        "target_label": "-",
        "grb_Status": "-",
        "grb_Runtime": "-",
        "grb_ObjectiveValue": "-",
        "grb_MIPGap": "-",
        "grb_NumVars": "-",
        "grb_NumBinVars": "-",
        "iteration_time": f"complete: {time_eval_duration}",
        "iteration_mem_consumption": f"complete: {mem_eval_consumption}",
    })

    pf = pd.DataFrame(results)
    pf.to_csv(ctf_output_path, index=False)


if __name__ == "__main__":
    ctf_evaluate()