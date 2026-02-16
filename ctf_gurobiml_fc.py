import tracemalloc

import onnx
from onnx2pytorch import ConvertModel
import gurobipy as gp
from gurobipy import GRB
from gurobi_ml import add_predictor_constr
import time
import gc
from utils import get_current_memory_mb
import torch.nn as nn
import config
import pandas as pd
from data import get_mnist_batch

def extract_layers(model):
    layers = []
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ReLU, nn.Sigmoid, nn.Tanh, nn.Flatten)):
            layers.append(module)
    return nn.Sequential(*layers)

def ctf_evaluate():
    # ================= Setup ===================
    onnx_path = config.FC_MNIST_PATH
    ctf_output_path = "results/final/gurobiml/ctf_fc_mnist.csv"
    n_inputs = 784
    n_outputs = 10
    iterations = config.ITERATIONS
    mnist_lb = config.MNIST_LB
    mnist_ub = config.MNIST_UB

    # ================= Load images and nn predictions
    images, labels = get_mnist_batch(iterations, flatten=True)
    ctf_targets = [0 for i in range(iterations)]
    results = []
    # ================ START ================

    tmp_constraints = []
    tmp_vars = []

    for i in range(iterations):
        gc.collect()
        tracemalloc.start()
        time_iteration_start = time.perf_counter()

        # ================= Gurobi ML ===================
        onnx_model = onnx.load(onnx_path)
        pytorch_model = ConvertModel(onnx_model)
        pytorch_model = extract_layers(pytorch_model)
        pytorch_model.eval()
        gurobi_model = gp.Model("Neural_Net_Optimization")
        gurobi_model.setParam("OutputFlag", 0)
        input_vars = gurobi_model.addMVar((1, n_inputs), lb=mnist_lb, ub=mnist_ub, name="input")
        output_vars = gurobi_model.addMVar((1, n_outputs), lb=-GRB.INFINITY, ub=+GRB.INFINITY, name="output")
        pred_constr = add_predictor_constr(gurobi_model, pytorch_model, input_vars, output_vars)
        gurobi_model.setParam("MIPFocus", config.MIPFOCUS)
        gurobi_model.setParam("TimeLimit", config.TIME_LIMIT)

        img = images[i].clone().flatten()
        label = labels[i]
        target = ctf_targets[i]

        # Define dist vars
        dist_vars = gurobi_model.addVars(n_inputs, name="dist_vars")
        input_vars_flat = input_vars[0].tolist()
        for idx, var in enumerate(input_vars_flat):
            gurobi_input_var = var
            dist_var = dist_vars[idx]
            # Realizing the L1 distance |x_var - x_original|
            c1 = gurobi_model.addConstr(dist_var >= img[idx] - gurobi_input_var)
            c2 = gurobi_model.addConstr(dist_var >= gurobi_input_var - img[idx])

        # Define constraints ensuring class 0 is predicted
        target_var = output_vars[0][target]
        for output_idx in range(output_vars.shape[1]):
            if target != output_idx:
                gurobi_output_var = output_vars[0][output_idx]
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