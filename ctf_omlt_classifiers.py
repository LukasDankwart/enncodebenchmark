import tracemalloc
from enum import Enum
import onnx
import pandas as pd
import torch
import numpy as np
import config
import gc
from utils import get_current_memory_mb
import time
import pyomo.environ as am
from omlt import OmltBlock
from omlt.io import write_onnx_model_with_bounds, load_onnx_neural_network_with_bounds
from omlt.neuralnet import FullSpaceNNFormulation

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

def reexport_onnx_with_bounds(input_path, output_path, lb, ub):
    model_proto = onnx.load(input_path)
    input_bounds = {}
    n_inputs = 11 if domain == Domain.WINE else 8
    for i in range(n_inputs):
        input_bounds[(i)] = (lb, ub)
    write_onnx_model_with_bounds(output_path, model_proto, input_bounds)


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
            ctf_output_path = "counterfactuals/omlt/classifiers/concrete/ctf_tiny_concrete.csv"
            output_node = "5"
        elif type == Type.MEDIUM:
            onnx_path = "models/datasets/trained/classifier_medium_concrete.onnx"
            ctf_output_path = "results/final/omlt/ctf_medium_concrete.csv"
            output_node = "11"
    elif domain == Domain.DIABETES:
        data_folder = "models/datasets/diabetes/split"
        MODEL_NAME = "diabetes"
        file_path_train = f'{data_folder}/diabetes_train.csv'
        file_path_test = f'{data_folder}/diabetes_test.csv'
        file_path_val = f'{data_folder}/diabetes_test.csv'
        if type == Type.TINY:
            onnx_path = "models/datasets/trained/classifier_tiny_diabetes.onnx"
            ctf_output_path = "counterfactuals/omlt/classifiers/diabetes/ctf_tiny_diabetes.csv"
            output_node = "5"
        elif type == Type.MEDIUM:
            onnx_path = "models/datasets/trained/classifier_medium_diabetes.onnx"
            ctf_output_path = "counterfactuals/omlt/classifiers/diabetes/ctf_medium_diabetes.csv"
            output_node = "11"
    elif domain == Domain.WINE:
        data_folder = "models/datasets/wine/split"
        MODEL_NAME = "wine"
        file_path_train = f'{data_folder}/winequality_train.csv'
        file_path_test = f'{data_folder}/winequality_test.csv'
        file_path_val = f'{data_folder}/winequality_val.csv'
        if type == Type.TINY:
            onnx_path = "models/datasets/trained/classifier_tiny_wine.onnx"
            ctf_output_path = "counterfactuals/omlt/classifiers/wine/ctf_tiny_wine.csv"
            output_node = "5"
        elif type == Type.MEDIUM:
            onnx_path = "models/datasets/trained/classifier_medium_wine.onnx"
            ctf_output_path = "results/final/omlt/ctf_medium_wine.csv"
            output_node = "11"
    elif domain == Domain.POWER:
        data_folder = "models/datasets/power/split"
        MODEL_NAME = "power"
        file_path_train = f'{data_folder}/train.csv'
        file_path_test = f'{data_folder}/test.csv'
        file_path_val = f'{data_folder}/val.csv'
        if type == Type.TINY:
            onnx_path = "models/datasets/trained/classifier_tiny_power.onnx"
            ctf_output_path = "counterfactuals/omlt/classifiers/power/ctf_tiny_power.csv"
            output_node = "5"
        elif type == Type.MEDIUM:
            onnx_path = "models/datasets/trained/classifier_medium_power.onnx"
            ctf_output_path = "counterfactuals/omlt/classifiers/power/ctf_medium_power.csv"
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

    # ================= Setup OMLT
    onnx_path_bounded = onnx_path.removesuffix(".onnx") + "_bounded.onnx"
    reexport_onnx_with_bounds(onnx_path, onnx_path_bounded, classifier_lb, classifier_ub)

    results = []

    # ================ START ================
    mem_eval_start = get_current_memory_mb()
    time_eval_start = time.perf_counter()

    gc.collect()
    for i in range(iterations):

        # Measurement for iteration starts after clean up
        gc.collect()
        tracemalloc.start()
        time_iteration_start = time.perf_counter()

        # OMLT to Pyomo setup
        network_definition = load_onnx_neural_network_with_bounds(onnx_path_bounded)
        model = am.ConcreteModel()
        model.nn = OmltBlock()
        formulation = FullSpaceNNFormulation(network_definition)
        model.nn.build_formulation(formulation)

        # Setup Solver and set timelimit and focus
        pyo_solver = am.SolverFactory('gurobi_persistent')
        pyo_solver.set_instance(model)
        native_model = pyo_solver._solver_model
        pyo_solver.options['TimeLimit'] = config.TIME_LIMIT
        pyo_solver.options['MIPFocus'] = config.MIPFOCUS
        pyo_input_vars = model.nn.inputs
        pyo_solver_input_keys = sorted(model.nn.inputs.keys())
        pyo_solver_output_keys = sorted(model.nn.outputs.keys())

        model.temp_block = am.Block()

        input_tens = X_test_tensor[i]
        target = torch.remainder(y_test_tensor[i] + 1, len(pyo_solver_output_keys))

        model.temp_block.dist_vars = am.Var(range(len(pyo_input_vars)))
        model.temp_block.dist_constraints = am.ConstraintList()

        for idx, input_var_key in enumerate(pyo_solver_input_keys):
            input_var = model.nn.inputs[input_var_key]
            d_var = model.temp_block.dist_vars[idx]
            val = input_tens[idx].item()

            # Realizing the L1 distance |x_var - x_original|
            model.temp_block.dist_constraints.add(d_var >= input_var - val)
            model.temp_block.dist_constraints.add(d_var >= val - input_var)

        model.temp_block.target_constraints = am.ConstraintList()
        target_var = model.nn.outputs[pyo_solver_output_keys[target]]

        # Define constraints ensuring target class is predicted
        for c_idx in range(len(pyo_solver_output_keys)):
            if c_idx != target:
                other_var = model.nn.outputs[pyo_solver_output_keys[c_idx]]
                model.temp_block.target_constraints.add(target_var >= other_var + 0.001)

        model.del_component(model.obj) if hasattr(model, 'obj') else None
        model.obj = am.Objective(expr=sum(model.temp_block.dist_vars[k] for k in model.temp_block.dist_vars),
                                 sense=am.minimize)

        pyo_solver.add_block(model.temp_block)
        pyo_solver.set_objective(model.obj)


        # ================= MEASUREMENTS ======================
        # Start time and memory measurement for the optimization

        solution = pyo_solver.solve(model, tee=False)

        # ================= Postprocessing ======================
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        mem_mb = current_mem / 10 ** 6

        time_iteration_end = time.perf_counter()
        time_duration = time_iteration_end - time_iteration_start

        print(f"Optimization of iteration {i} is finished!")

        model.dummy_obj = am.Objective(expr=0)
        pyo_solver.set_objective(model.dummy_obj)

        pyo_solver.remove_block(model.temp_block)
        model.del_component(model.temp_block)
        model.del_component(model.dummy_obj)
        if hasattr(model, 'obj'):
            model.del_component(model.obj)

        if hasattr(model, 'temp_block'):
            pyo_solver.remove_block(model.temp_block)
            model.del_component(model.temp_block)
            del model.temp_block

        try:
            gap = native_model.MIPGap
        except AttributeError:
            gap = "Not accessible"

        results.append({
            "iteration": i,
            "gt_label": y_test_tensor[i].item(),
            "target_label": target.item(),
            "grb_Status": native_model.status,
            "grb_Runtime": native_model.runtime,
            "grb_ObjectiveValue": native_model.objVal,
            "grb_MIPGap": gap,
            "grb_NumVars": native_model.numVars,
            "grb_NumBinVars": native_model.numBinVars,
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