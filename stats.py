import pandas as pd
from enum import Enum
import numpy as np

import config


class Variant(Enum):
    FC = 0
    CONV = 1
    CONCRETE = 2
    WINE = 3

def compute_stats():

    if variant == Variant.FC:
        paths = [
            "results/final/onnxgurobi/ctf_fc_mnist.csv",
            "results/final/omlt/ctf_fc_mnist.csv",
            "results/final/gurobiml/ctf_fc_mnist.csv"
        ]
        ctf_output_path = "results/final/fc_comparison.csv"
    if variant == Variant.CONV:
        paths = [
            "results/final/onnxgurobi/ctf_conv_mnist.csv",
            "results/final/omlt/ctf_conv_mnist.csv"
        ]
        ctf_output_path = "results/final/conv_comparison.csv"
    if variant == Variant.CONCRETE:
        paths = [
            "results/final/onnxgurobi/ctf_medium_concrete.csv",
            "results/final/omlt/ctf_medium_concrete.csv",
            "results/final/gurobiml/ctf_medium_concrete.csv"
        ]
        ctf_output_path = "results/final/concrete_comparison.csv"
    if variant == Variant.WINE:
        paths = [
            "results/final/onnxgurobi/ctf_medium_wine.csv",
            "results/final/omlt/ctf_medium_wine.csv",
            "results/final/gurobiml/ctf_medium_wine.csv"
        ]
        ctf_output_path = "results/final/wine_comparison.csv"

    results = []
    for path in paths:

        df = pd.read_csv(path)

        path_name = path.split("/")[-1].removesuffix(".csv")
        variant_name = path.split("/")[-2]

        colum_names = ["iteration_time", "iteration_mem_consumption"]

        for col in colum_names:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df_clean = df[df["iteration_time"] <= config.TIME_LIMIT]

        mean_series = df_clean[colum_names].mean()
        std_series = df_clean[colum_names].std()

        results.append({
            "variant_name": variant_name,
            "path_name": path_name,

            "time_mean": mean_series["iteration_time"],
            "time_std": std_series["iteration_time"],

            "mem_mean": mean_series["iteration_mem_consumption"],
            "mem_std": std_series["iteration_mem_consumption"]
        })

    pf = pd.DataFrame(results)
    pf.to_csv(ctf_output_path, index=False)


def compress_results():
    paths = [
        "results/final/fc_comparison.csv",
        "results/final/conv_comparison.csv",
        "results/final/concrete_comparison.csv",
        "results/final/wine_comparison.csv"
    ]
    collection = []
    for file in paths:
        df = pd.read_csv(file)
        collection.append(df)
        spacer = pd.DataFrame("-", index=[0, 1], columns=df.columns)
        collection.append(spacer)

    compression = pd.concat(collection[:-1], ignore_index=True)
    compression.to_csv("results/final/overview.csv", index=False, na_rep="")

if __name__ == "__main__":
    variant = Variant.CONCRETE
    compute_stats()

    compress_results()