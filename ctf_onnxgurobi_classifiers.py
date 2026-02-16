import tracemalloc
from enum import Enum
import onnx
import pandas as pd
import torch
import numpy as np
import config
from onnx_to_gurobi.onnxToGurobi import ONNXToGurobi
import gc
from gurobipy import GRB
from utils import get_current_memory_mb
import time

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
        if type == Type.TINY:
            onnx_path = "models/datasets/trained/classifier_tiny_concrete.onnx"
            ctf_output_path = "counterfactuals/onnxgurobi/classifiers/concrete/ctf_tiny_concrete.csv"
            output_node = "5"
        elif type == Type.MEDIUM:
            onnx_path = "models/datasets/trained/classifier_medium_concrete.onnx"
            ctf_output_path = "results/final/onnxgurobi/ctf_medium_concrete.csv"
            output_node = "11"
    elif domain == Domain.DIABETES:
        data_folder = "models/datasets/diabetes/split"
        MODEL_NAME = "diabetes"
        file_path_train = f'{data_folder}/diabetes_train.csv'
        file_path_test = f'{data_folder}/diabetes_test.csv'
        file_path_val = f'{data_folder}/diabetes_test.csv'
        if type == Type.TINY:
            onnx_path = "models/datasets/trained/classifier_tiny_diabetes.onnx"
            ctf_output_path = "counterfactuals/onnxgurobi/classifiers/diabetes/ctf_tiny_diabetes.csv"
            output_node = "5"
        elif type == Type.MEDIUM:
            onnx_path = "models/datasets/trained/classifier_medium_diabetes.onnx"
            ctf_output_path = "counterfactuals/onnxgurobi/classifiers/diabetes/ctf_medium_diabetes.csv"
            output_node = "11"
    elif domain == Domain.WINE:
        data_folder = "models/datasets/wine/split"
        MODEL_NAME = "wine"
        file_path_train = f'{data_folder}/winequality_train.csv'
        file_path_test = f'{data_folder}/winequality_test.csv'
        file_path_val = f'{data_folder}/winequality_val.csv'
        if type == Type.TINY:
            onnx_path = "models/datasets/trained/classifier_tiny_wine.onnx"
            ctf_output_path = "counterfactuals/onnxgurobi/classifiers/wine/ctf_tiny_wine.csv"
            output_node = "5"
        elif type == Type.MEDIUM:
            onnx_path = "models/datasets/trained/classifier_medium_wine.onnx"
            ctf_output_path = "results/final/onnxgurobi/ctf_medium_wine.csv"
            output_node = "11"
    elif domain == Domain.POWER:
        data_folder = "models/datasets/power/split"
        MODEL_NAME = "power"
        file_path_train = f'{data_folder}/train.csv'
        file_path_test = f'{data_folder}/test.csv'
        file_path_val = f'{data_folder}/val.csv'
        if type == Type.TINY:
            onnx_path = "models/datasets/trained/classifier_tiny_power.onnx"
            ctf_output_path = "counterfactuals/onnxgurobi/classifiers/power/ctf_tiny_power.csv"
            output_node = "5"
        elif type == Type.MEDIUM:
            onnx_path = "models/datasets/trained/classifier_medium_power.onnx"
            ctf_output_path = "counterfactuals/onnxgurobi/classifiers/power/ctf_medium_power.csv"
            output_node = "11"

    # Inputs for CTF evaluation are selected from the test datasets
    df_test = pd.read_csv(file_path_test, delimiter=',')

    X_test = df_test.drop(columns="target")
    y_test = df_test["target"]

    X_test_tensor = torch.tensor(X_test.to_numpy().astype(np.float32))
    y_test_tensor = torch.tensor(y_test.to_numpy().astype(np.int64))


    # ================= Setup ===================
    iterations = config.ITERATIONS
    classifier_lb = config.CLASSIFIERS_LB
    classifier_ub = config.CLASSIFIERS_UB


    results = []

    # ================ START ================
    mem_eval_start = get_current_memory_mb()
    time_eval_start = time.perf_counter()

    tmp_constraints = []
    tmp_vars = []
    gc.collect()
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

        input_tens = X_test_tensor[i]
        target = torch.remainder(y_test_tensor[i] + 1, len(gurobi_model_output_vars))

        # Define dist vars
        dist_vars = gurobi_model.addVars(len(gurobi_model_input_vars), name="dist_vars")
        tmp_vars.append(dist_vars)

        for flat_idx, key_tuple in enumerate(gurobi_input_keys):
            gurobi_input_var = gurobi_model_input_vars[key_tuple]
            dist_var = dist_vars[flat_idx]
            # Realizing the L1 distance |x_var - x_original|
            c1 = gurobi_model.addConstr(dist_var >= input_tens[flat_idx] - gurobi_input_var)
            c2 = gurobi_model.addConstr(dist_var >= gurobi_input_var - input_tens[flat_idx])
            tmp_constraints.extend([c1, c2])


        # Define constraints ensuring CTF target is test_target idx + 1
        target_var = gurobi_model_output_vars.get(gurobi_output_keys[target])
        for output_idx in range(len(gurobi_model_output_vars)):
            if target != output_idx:
                gurobi_output_var = gurobi_model_output_vars.get(gurobi_output_keys[output_idx])
                c_target = gurobi_model.addConstr(target_var >= gurobi_output_var + 0.001)
                tmp_constraints.extend([c_target])

        # Set objective to minimize L1 distance btw. input and ctf
        gurobi_model.setObjective(dist_vars.sum(), GRB.MINIMIZE)


        # ================= MEASUREMENTS ======================

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
            "gt_label": y_test_tensor[i].item(),
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
    # Experiments for CONCRETE
    domain = Domain.WINE
    type = Type.MEDIUM
    ctf_evaluation()
