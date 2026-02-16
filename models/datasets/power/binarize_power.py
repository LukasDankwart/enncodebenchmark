import pandas as pd


def binarize_target(input_file="power.csv", output_file="power_binarized.csv"):
    # Load the dataset
    data = pd.read_csv(input_file)

    # Assume the last column is the target
    target_column = data.columns[-1]

    # Calculate the mean of the target column
    mean_value = data[target_column].mean()

    # Binarize the target column: 1 if >= mean, 0 otherwise
    data[target_column] = (data[target_column] >= mean_value).astype(int)

    # Save the modified dataset to a new CSV file
    data.to_csv(output_file, index=False)
    print(f"Binarized dataset saved to '{output_file}'")


if __name__ == '__main__':
    # Run the function
    binarize_target()
