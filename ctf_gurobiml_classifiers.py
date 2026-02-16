import tracemalloc

import torch
import onnx
from onnx2pytorch import ConvertModel
import gurobipy as gp
from gurobipy import GRB
from gurobi_ml import add_predictor_constr
import time
import gc
from utils import get_current_memory_mb
import torch.nn as nn
from enum import Enum
import config
import pandas as pd
import numpy as np

def extract_layers(model):
    layers = []
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ReLU, nn.Sigmoid, nn.Tanh, nn.Flatten)):
            layers.append(module)
    return nn.Sequential(*layers)


class Domain(Enum):
    CONCRETE = 1
    DIABETES = 2
    CANCER = 3
    WINE = 4
    MOONS = 5
    FETAL = 6
    POWER = 7


class Type(Enum):
    TINY = 1
    MEDIUM = 2


def ctf_evaluation():
    input_node = "onnx::MatMul_0"
    # Load the dataset from a CSV file
    if domain == Domain.CONCRETE:
        data_folder = "models/datasets/concrete/split"
        MODEL_NAME = "concrete"
        file_path_train = f'{data_folder}/concrete_train.csv'
        file_path_test = f'{data_folder}/concrete_test.csv'
        file_path_val = f'{data_folder}/concrete_val.csv'
        n_inputs = 8
        n_outputs = 2
        if type == Type.TINY:
            onnx_path = "models/datasets/trained/classifier_tiny_concrete.onnx"
            ctf_output_path = "counterfactuals/gurobiml/classifiers/concrete/ctf_tiny_concrete.csv"
            output_node = "5"
        elif type == Type.MEDIUM:
            onnx_path = "models/datasets/trained/classifier_medium_concrete.onnx"
            ctf_output_path = "results/final/gurobiml/ctf_medium_concrete.csv"
            output_node = "11"
    elif domain == Domain.DIABETES:
        data_folder = "models/datasets/diabetes/split"
        MODEL_NAME = "diabetes"
        file_path_train = f'{data_folder}/diabetes_train.csv'
        file_path_test = f'{data_folder}/diabetes_test.csv'
        file_path_val = f'{data_folder}/diabetes_test.csv'
        n_inputs = 8
        n_outputs = 2
        if type == Type.TINY:
            onnx_path = "models/datasets/trained/classifier_tiny_diabetes.onnx"
            ctf_output_path = "counterfactuals/gurobiml/classifiers/diabetes/ctf_tiny_diabetes.csv"
            output_node = "5"
        elif type == Type.MEDIUM:
            onnx_path = "models/datasets/trained/classifier_medium_diabetes.onnx"
            ctf_output_path = "counterfactuals/gurobiml/classifiers/diabetes/ctf_medium_diabetes.csv"
            output_node = "11"
    elif domain == Domain.WINE:
        data_folder = "models/datasets/wine/split"
        MODEL_NAME = "wine"
        file_path_train = f'{data_folder}/winequality_train.csv'
        file_path_test = f'{data_folder}/winequality_test.csv'
        file_path_val = f'{data_folder}/winequality_val.csv'
        n_inputs = 11
        n_outputs = 2
        if type == Type.TINY:
            onnx_path = "models/datasets/trained/classifier_tiny_wine.onnx"
            ctf_output_path = "counterfactuals/gurobiml/classifiers/wine/ctf_tiny_wine.csv"
            output_node = "5"
        elif type == Type.MEDIUM:
            onnx_path = "models/datasets/trained/classifier_medium_wine.onnx"
            ctf_output_path = "results/final/gurobiml/ctf_medium_wine.csv"
            output_node = "11"
    elif domain == Domain.POWER:
        data_folder = "models/datasets/power/split"
        MODEL_NAME = "power"
        file_path_train = f'{data_folder}/train.csv'
        file_path_test = f'{data_folder}/test.csv'
        file_path_val = f'{data_folder}/val.csv'
        n_inputs = 4
        n_outputs = 2
        if type == Type.TINY:
            onnx_path = "models/datasets/trained/classifier_tiny_power.onnx"
            ctf_output_path = "counterfactuals/gurobiml/classifiers/power/ctf_tiny_power.csv"
            output_node = "5"
        elif type == Type.MEDIUM:
            onnx_path = "models/datasets/trained/classifier_medium_power.onnx"
            ctf_output_path = "counterfactuals/gurobiml/classifiers/power/ctf_medium_power.csv"
            output_node = "11"

    # Inputs for CTF evaluation are selected from the test datasets
    df_test = pd.read_csv(file_path_test, delimiter=',')

    X_test = df_test.drop(columns="target")
    y_test = df_test["target"]

    X_test_tensor = torch.tensor(X_test.to_numpy().astype(np.float32))
    y_test_tensor = torch.tensor(y_test.to_numpy().astype(np.int64))


    # ================= Setup ===================
    iterations = config.ITERATIONS
    classifiers_lb = config.CLASSIFIERS_LB
    classifiers_ub = config.CLASSIFIERS_UB



    results = []
    # ================ START ================

    tmp_constraints = []
    tmp_vars = []
    gc.collect()
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
        input_vars = gurobi_model.addMVar((1, n_inputs), lb=classifiers_lb, ub=classifiers_ub, name="input")
        output_vars = gurobi_model.addMVar((1, n_outputs), lb=-GRB.INFINITY, ub=+GRB.INFINITY, name="output")
        pred_constr = add_predictor_constr(gurobi_model, pytorch_model, input_vars, output_vars)
        gurobi_model.setParam("MIPFocus", config.MIPFOCUS)
        gurobi_model.setParam("TimeLimit", config.TIME_LIMIT)

        input_tens = X_test_tensor[i]
        target = torch.remainder(y_test_tensor[i] + 1, output_vars.shape[1])

        # Define dist vars
        dist_vars = gurobi_model.addVars(input_vars.shape[1], name="dist_vars")
        tmp_vars.append(dist_vars)

        for idx, var in enumerate(input_vars):
            dist_var = dist_vars[idx]
            # Realizing the L1 distance |x_var - x_original|
            c1 = gurobi_model.addConstr(dist_var >= input_tens[idx] - var)
            c2 = gurobi_model.addConstr(dist_var >= var - input_tens[idx])
            tmp_constraints.extend([c1, c2])

        target_var = output_vars[0][target.item()]
        # Define constraints ensuring CTF target is test_target idx + 1
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
        print(f"Optimization of iteration {i} is finished!")

        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        mem_mb = current_mem / 10 ** 6

        time_iteration_end = time.perf_counter()
        time_duration = time_iteration_end - time_iteration_start

        try:
            gap = gurobi_model.MIPGap
        except AttributeError:
            gap = "Not accessible"

        results.append({
            "iteration": i,
            "gt_label": y_test_tensor[i].item(),
            "target_label": target.item(),
            "grb_Status": gurobi_model.status,
            "grb_Runtime": gurobi_model.runtime,
            "grb_ObjectiveValue": gurobi_model.objVal if gurobi_model.SolCount > 0 else None,
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
    # Experiments for CONCRETE
    domain = Domain.WINE
    type = Type.MEDIUM
    ctf_evaluation()