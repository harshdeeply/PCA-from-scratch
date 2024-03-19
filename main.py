import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    # Reading the dataset and loading it in pandas dataframe
    df = pd.read_csv(file_path, skiprows=[0], sep=',', names=['sepal_length_cm', 'sepal_width_cm', 'petal_length_cm', 'petal_width_cm', 'species'])
    return df

def normalize_data(data):
    # Separating feature attributes and species attribute
    features_df = data.loc[:, ['sepal_length_cm', 'sepal_width_cm', 'petal_length_cm', 'petal_width_cm']]

    # Normalizing feature attribute data
    features_df = StandardScaler().fit_transform(features_df)
    return features_df

def calculate_covariance_matrix(data):
    # Calculating mean of column values (features)
    mean_vals = np.mean(data, axis=0)

    # Calculating covariance of the feature matrix by subtracting the mean values
    covariance_matrix = (data - mean_vals).T.dot((data - mean_vals)) / (data.shape[0] - 1)
    return covariance_matrix

def calculate_principal_components(covariance_matrix):
    # Calculating Eigenvalues and Eigenvectors which satisfy the equation
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Creating a pair of eigenvalues and eigenvectors and sorting them in descending order
    eigen_pairs = [(np.abs(eigenvalues[i]), eigenvectors[:, i]) for i in range(len(eigenvalues))]
    eigen_pairs.sort()
    eigen_pairs.reverse()

    # Extracting top 2 features that have max eigenvalues
    top_2_matrix = np.hstack((eigen_pairs[0][1].reshape(4, 1), eigen_pairs[1][1].reshape(4, 1)))
    return top_2_matrix

def plot_pca_results(data, pca_data):
    # Creating a new pandas dataframe and using new pca attribute values generated after transformation
    pca_df = pd.DataFrame(data=pca_data, columns=['pca1', 'pca2'])
    new_df = pd.concat([pca_df, data[['species']]], axis=1)  # axis = 1 is for columns

    # Plotting the graph
    fig = plt.figure(figsize=(7, 7))
    axis = fig.add_subplot(111, facecolor='white')
    axis.set_xlabel('PCA 1', fontsize=12)
    axis.set_ylabel('PCA 2', fontsize=12)
    axis.set_title('PCA on Iris dataset', fontsize=15)

    classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    for species_val, color in zip(classes, ['y', 'b', 'r']):
        index_vals = new_df['species'] == species_val
        axis.scatter(new_df.loc[index_vals, 'pca1'], new_df.loc[index_vals, 'pca2'], c=color, s=50)
    axis.legend(classes, loc="upper right")
    axis.grid(linewidth=0.5)
    fig.savefig('pca-output.png', dpi=300)

def main():
    # Load data
    file_path = 'iris.csv'
    data = load_data(file_path)

    # Normalize data
    normalized_data = normalize_data(data)

    # Calculate covariance matrix
    covariance_matrix = calculate_covariance_matrix(normalized_data)

    # Calculate principal components
    principal_components = calculate_principal_components(covariance_matrix)

    # Generate top 2 principal components
    pca_result = normalized_data.dot(principal_components)

    # Plot PCA results
    plot_pca_results(data, pca_result)

if __name__ == "__main__":
    main()
