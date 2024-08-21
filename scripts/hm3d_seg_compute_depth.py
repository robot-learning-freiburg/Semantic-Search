from sem_objnav.segment.hm3d.dataset import Hm3d

dataset = Hm3d(
    dataset_path="../calibration_dataset/",
    split="train",
)
print(dataset.depth_compute_stats())
# {'mean': 2428.05730289446, 'std': 1061.8438682864453, 'min': 280.0, 'max': 5599.0} dataset without objects
# {'mean': 1951.0407430546513, 'std': 1060.2024147548186, 'min': 500.0, 'max': 5000.0} with objects
# {'mean': 1943.3607361343045, 'std': 1064.2138822762122, 'min': 500.0, 'max': 5000.0} with objects updated
# hm3d34 {'mean': 1917.436629394096, 'std': 1069.5006390726933, 'min': 500.0, 'max': 5000.0}
