import onnx
import psutil
import os
import random
import ast
import numpy as np
import onnxruntime as ort

def run_onnx_on_img(path, img):
    ort_session = ort.InferenceSession(path)
    input_name = ort_session.get_inputs()[0].name
    outputs = ort_session.run(None, {input_name: img})
    logits = outputs[0]
    predicted_class = np.argmax(logits, axis=1)[0]
    return predicted_class

def get_current_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def generate_conditions(path, num_outputs, num_conditions=10):
    random.seed(5)
    test_constraints = [(random.randint(0, num_outputs-1), random.uniform(-0.5, 0.5)) for _ in range(num_conditions)]
    with open(path, "w", encoding="utf-8") as f:
        for constraint in test_constraints:
            f.write(f"{constraint}\n")


def load_conditions(path):
    if os.path.isfile(path):
        with open(path, 'r') as file:
            lines = file.readlines()
            test_conditions = []
            for line in lines:
                line = line.strip()
                if not line: continue
                try:
                    condition = ast.literal_eval(line)
                    test_conditions.append(condition)
                except (ValueError, SyntaxError):
                    print(f"Error while parsing line: {line}")
            return np.array(test_conditions)
    else:
        raise RuntimeError(f"Given path {path} is no valid .txt file.")


def make_dim_dynamic(tensor_list, dim_index=0, param_name="batch_size"):
    for tensor in tensor_list:
        shape = tensor.type.tensor_type.shape

        if len(shape.dim) > dim_index:
            target_dim = shape.dim[dim_index]


            if target_dim.HasField("dim_value"):
                target_dim.ClearField("dim_value")

            target_dim.dim_param = param_name

            print(f"Updated '{tensor.name}': Dim {dim_index} ist jetzt dynamisch ('{param_name}').")


def generate_original_inputs(output_path, shape, nums=10):
    random.seed(5)
    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(nums):
            inp = np.random.random(shape)
            f.write(str(inp) + "\n")


def add_dynamic_batch_dimension(path):
    output_path = path.removesuffix(".onnx") + "_modified.onnx"
    model = onnx.load(path)
    graph = model.graph
    make_dim_dynamic(graph.input)
    make_dim_dynamic(graph.output)
    model.graph.value_info.clear()
    onnx.checker.check_model(model)
    onnx.save(model, output_path)


def load_original_inputs(path):
    inputs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            clean_line = line.replace('[', '').replace(']', '')
            if not clean_line.strip():
                continue
            numbers = [float(x) for x in clean_line.split()]
            inputs.append(numbers)
    inputs = np.array(inputs)
    return inputs


def load_large_inputs(path, shape):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    clean_content = content.replace('[', '').replace(']', '')

    data_flat = np.array([float(x) for x in clean_content.split()])
    total_nums = data_flat.size

    features_per_sample = np.prod(shape)

    if features_per_sample == 0:
        raise ValueError("Sample Shape darf nicht 0 enthalten!")

    if total_nums % features_per_sample != 0:
        print(f"WARNUNG: Gesamtanzahl Zahlen ({total_nums}) ist nicht durch {features_per_sample} teilbar!")
        print(f"Es könnte sein, dass das letzte Sample abgeschnitten ist.")

    target_shape = (-1,) + tuple(shape)
    inputs_tensor = data_flat.reshape(target_shape)

    return inputs_tensor



if __name__ == "__main__":
    shape = (1, 1)
    path = "models/nn4sys/inputs_lindexdeep.txt"
    generate_original_inputs(path, shape=shape)
    load_large_inputs(path, shape=shape)
    cond_path = "models/nn4sys/conditions_lindexdeep.txt"
    generate_conditions(cond_path, num_outputs=1)