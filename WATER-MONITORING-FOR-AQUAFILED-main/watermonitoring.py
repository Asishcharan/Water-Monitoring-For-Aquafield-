import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial import distance
import warnings

# Define the reference colors with floating-point pH values
reference_colors = {
    (255, 0, 0): 1.0,  # Red
    (255, 69, 0): 1.5,  # Reddish-orange
    (255, 165, 0): 2.0,  # Orange
    (255, 215, 0): 2.5,  # Gold
    (255, 255, 0): 3.0,  # Yellow
    (238, 232, 170): 3.5,  # Pale yellow
    (173, 255, 47): 4.0,  # Yellow-green
    (127, 255, 0): 4.5,  # Lime
    (0, 255, 0): 5.0,  # Green
    (34, 139, 34): 5.5,  # Forest green
    (0, 128, 0): 6.0,  # Dark green
    (0, 206, 209): 6.5,  # Turquoise
    (0, 255, 255): 7.0,  # Cyan
    (0, 191, 255): 7.5,  # Deep sky blue
    (0, 0, 255): 8.0,  # Blue
    (75, 0, 130): 9.0,  # Indigo
    (148, 0, 211): 10.0,  # Violet
    (128, 0, 128): 11.0,  # Purple
    (255, 105, 180): 12.0,  # Pink
    (255, 20, 147): 13.0,  # Deep pink
    (139, 0, 0): 14.0  # Dark red
}


# Function to read image
def read_image(file_path):
    image = cv2.imread(file_path)
    if image is None:
        raise ValueError("Image not found or path is incorrect")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


# Function to get dominant color using KMeans clustering
def get_dominant_color_kmeans(image, k=5):
    data = image.reshape((-1, 3))
    data = np.float32(data)
    unique_colors = np.unique(data, axis=0)
    k = min(len(unique_colors), k)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)

    centers = np.array(kmeans.cluster_centers_, dtype='uint8')
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    dominant_color = centers[labels[np.argmax(counts)]]
    return tuple(dominant_color)


# Function to get the exact or closest pH value
def get_closest_ph(dominant_color, reference_colors):
    distances = []
    for ref_color, ph_value in reference_colors.items():
        dist = distance.euclidean(dominant_color, ref_color)
        distances.append((dist, ph_value))
    closest_ph = min(distances, key=lambda x: x[0])[1]
    return closest_ph


# Load the real dataset from CSV
def load_fish_dataset(csv_file):
    return pd.read_csv(csv_file)


# Find suitable fish for the given pH value from the dataset
def get_suitable_fish_for_ph(ph_value, fish_data, tolerance=0.05):
    suitable_fish = fish_data[np.isclose(fish_data['ph'], ph_value, atol=tolerance)]
    return suitable_fish['fish'].tolist() if not suitable_fish.empty else [
        "No suitable fish found for this pH level."]


if __name__ == "__main__":
    # Path to image and dataset
    file_path = r"D:\WATER-MONITORING-FOR-AQUAFILED-main\pond.jpg"
    csv_file = r"D:\WATER-MONITORING-FOR-AQUAFILED-main\realfishdataset.csv"  # Provide your real dataset path here

    # Load fish dataset
    fish_data = load_fish_dataset(csv_file)

    # Read image and get dominant color using K-means
    try:
        image = read_image(file_path)
        dominant_color = get_dominant_color_kmeans(image)

        # Get the exact or closest pH value to the dominant color
        closest_ph = get_closest_ph(dominant_color, reference_colors)

        print(f"Dominant color (RGB): {dominant_color}")
        print(f"Closest pH value: {closest_ph:.1f}")

        # Find suitable fish species for the determined pH value from the dataset
        suitable_fish = get_suitable_fish_for_ph(closest_ph, fish_data, tolerance=0.05)
        print(f"Suitable fish for pH {closest_ph:.1f}: {', '.join(suitable_fish)}")

    except Exception as e:
        print(f"Error: {e}")
