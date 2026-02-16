import time
import tracemalloc

import pyomo.environ as am
from omlt import OmltBlock
from omlt.io import write_onnx_model_with_bounds, load_onnx_neural_network_with_bounds
from omlt.neuralnet import FullSpaceNNFormulation
import onnx
import numpy as np
import gc
from utils import get_current_memory_mb, run_onnx_on_img
from data import get_mnist_batch
import config
import cv2
import pandas as pd

def reexport_onnx_with_bounds(input_path, output_path, lb, ub):
    model_proto = onnx.load(input_path)
    input_bounds = {}
    for i in range(784):
        input_bounds[(i)] = (lb, ub)

    write_onnx_model_with_bounds(output_path, model_proto, input_bounds)


def ctf_evaluate():
    # ================= Setup ===================
    onnx_path = config.FC_MNIST_PATH
    onnx_path_bounded = config.FC_MNIST_PATH_BOUNDS
    iterations = config.ITERATIONS
    mnist_lb = config.MNIST_LB
    mnist_ub = config.MNIST_UB
    reexport_onnx_with_bounds(onnx_path, onnx_path_bounded, mnist_lb, mnist_ub)
    ctf_output_path = "results/final/omlt/ctf_fc_mnist.csv"

    # ================= Setup OMLT

    images, labels = get_mnist_batch(iterations, flatten=True)
    ctf_targets = [0 for i in range(iterations)]
    results = []

    # ================ START ================
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

        img = images[i].clone().flatten()
        label = labels[i]
        target = ctf_targets[i]

        model.temp_block.dist_vars = am.Var(range(len(pyo_input_vars)))
        model.temp_block.dist_constraints = am.ConstraintList()

        for idx, input_var_key in enumerate(pyo_solver_input_keys):
            input_var = model.nn.inputs[input_var_key]
            d_var = model.temp_block.dist_vars[idx]
            val = img[idx].item()

            # Realizing the L1 distance |x_var - x_original|
            model.temp_block.dist_constraints.add(d_var >= input_var - val)
            model.temp_block.dist_constraints.add(d_var >= val - input_var)

        model.temp_block.target_constraints = am.ConstraintList()
        target_var = model.nn.outputs[pyo_solver_output_keys[target]]

        # Define constraints ensuring target class is predicted
        for c_idx in range(10):
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
            "gt_label": label,
            "target_label": target,
            "grb_Status": native_model.status,
            "grb_Runtime": native_model.runtime,
            "grb_ObjectiveValue": gap,
            "grb_MIPGap": native_model.MIPGap,
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
    ctf_evaluate()