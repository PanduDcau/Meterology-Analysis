import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm

# Configuration
DATA_FOLDER = 'data/'  # Folder containing your city files
N_CLASSES = 5  # Number of classes
N_NEIGHBORS = 5  # Number of neighbors for KNN
RANDOM_STATE = 42  # For reproducibility


def load_city_data(file_path):
    """Load and process a single city file"""
    ds = xr.open_dataset(file_path)

    # Try to automatically detect time dimension
    time_dim = next((dim for dim in ['time', 'valid_time', 'Time', 'TIME']
                     if dim in ds.dims), None)

    if time_dim is None:
        raise ValueError(f"Cannot find time dimension. Available dimensions: {list(ds.dims)}")

    # Extract relevant variables (modify as needed)
    variables = []
    var_names = []
    for var in ['t2m', 'sp', 'rh']:  # Example variables - adjust to match your data
        if var in ds:
            try:
                var_data = ds[var].mean(dim=time_dim).values  # Temporal mean
                variables.append(var_data)
                var_names.append(var)
            except Exception as e:
                print(f"Could not process variable {var}: {str(e)}")

    # Flatten and combine all variables
    feature_vector = np.concatenate([v.ravel() for v in variables])
    city_name = os.path.basename(file_path).split('_')[0]

    return city_name, feature_vector, var_names


def main():
    # 1. Load all city data
    print("Loading city data...")
    city_data = []
    city_names = []

    # Get all files in data folder
    files = sorted([f for f in os.listdir(DATA_FOLDER) if f.endswith('.nc')])
    assert len(files) == 34, f"Expected 34 files, found {len(files)}"

    for file in tqdm(files):
        try:
            city_name, features, _ = load_city_data(os.path.join(DATA_FOLDER, file))
            city_data.append(features)
            city_names.append(city_name)
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            continue

    # 2. Prepare data for classification
    X = np.array(city_data)
    y = np.random.randint(0, N_CLASSES, size=len(city_names))  # Replace with your actual labels

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Perform KNN classification
    print("\nPerforming KNN classification...")
    knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
    knn.fit(X_scaled, y)
    y_pred = knn.predict(X_scaled)

    # 4. Create city-cluster table
    results_df = pd.DataFrame({
        'City': city_names,
        'Cluster': y_pred
    }).sort_values('Cluster')

    print("\nCity to Cluster Assignment:")
    print(results_df.to_string(index=False))

    # 5. Visualize results with PCA
    print("\nVisualizing results...")
    plt.figure(figsize=(14, 8))

    # Create a grid for two plots
    grid = plt.GridSpec(1, 2, width_ratios=[2, 1])

    # PCA Plot
    ax1 = plt.subplot(grid[0])
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap='viridis', s=100)

    # Annotate points with city names
    for i, name in enumerate(city_names):
        ax1.annotate(name, (X_pca[i, 0], X_pca[i, 1]), fontsize=8, alpha=0.7)

    ax1.set_xlabel('Principal Component 1')
    ax1.set_ylabel('Principal Component 2')
    ax1.set_title(f'City Clusters (KNN, k={N_NEIGHBORS})')
    ax1.grid(True)

    # Cluster Legend
    ax2 = plt.subplot(grid[1])
    ax2.axis('off')

    cluster_info = []
    for cluster in range(N_CLASSES):
        cities = results_df[results_df['Cluster'] == cluster]['City'].tolist()
        cluster_info.append(f"Cluster {cluster}:\n" + "\n".join(cities))

    ax2.text(0.1, 0.5, "\n\n".join(cluster_info),
             va='center', ha='left', fontsize=9,
             bbox=dict(facecolor='white', alpha=0.7))

    plt.tight_layout()
    plt.show()

    # 6. Save results
    results_df.to_csv('city_clusters.csv', index=False)
    print("\nResults saved to 'city_clusters.csv'")


if __name__ == "__main__":
    main()