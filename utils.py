import pandas as pd

def get_df(csv_path, column_names):
    df = pd.read_csv(csv_path, header = None, names = column_names)
    return df

def show_image(df, image_directory):
    # Select a value from the first column of the DataFrame
    selected_row = dataframe.sample(1)
    selected_value = selected_row.iloc[0, 0]

    # Construct the image file path
    image_filename = os.path.join(image_folder, selected_value)

    # Read and plot the image using Matplotlib
    image = plt.imread(image_filename)
    plt.imshow(image)
    plt.title(f"Image from Row {selected_row.index[0]}")
    plt.show()