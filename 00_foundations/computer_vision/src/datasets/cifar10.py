
from pathlib import Path
import numpy as np

# ======== XX ======== || ======== XX ========
# Load CIFAR10
# ======== XX ======== || ======== XX ========


def load_cifar10_batch(file_path:Path):

    if file_path.is_file():
        
        with open(file_path, "rb") as f:
            import pickle
            batch = pickle.load(f, encoding='bytes')
            batch = {k.decode('utf-8'):v for k, v in batch.items()}
        
        B = batch["data"].shape[0]
        X = batch["data"].reshape(B, 3, 32, 32)
        Y = batch["labels"]
        return X, Y
    
    return None, None


def load_cifar10(folder_path:Path):
    if folder_path.is_dir():

        xtr, ytr = [], []
        # Iterate over train_data_batches
        for i in range(1, 6):
            file_path = folder_path / f"data_batch_{i}"
            X_batch, Y_batch = load_cifar10_batch(file_path)
            xtr.append(X_batch)
            ytr = ytr + Y_batch

        # Train batches: Concatenate
        Xtr = np.concatenate(xtr, axis=0)
        Ytr = np.array(ytr)

        # Iterate over train_data_batches
        file_path = folder_path / f"test_batch"
        Xts, Yts = load_cifar10_batch(file_path)
        Yts = np.array(Yts)

        # Get label idx to name mapping
        mapping_file_path = folder_path / "batches.meta"
        import pickle
        with open(mapping_file_path, "rb") as f:
            label_map = pickle.load(f, encoding='bytes')
            label_map = {k.decode('utf-8'):v for k, v in label_map.items()}
            label_map["label_names"] = [name.decode('utf-8') for name in label_map["label_names"]]
        
        
        return Xtr, Ytr, Xts, Yts, label_map
    
    return None, None, None, None, None



if __name__ == "__main__":

    cv_data_root_path = Path("/Users/irabandutta/Developer/2026-06-deep-learning-workbench/00_foundations/computer_vision/data")
    cifar10_path = cv_data_root_path / "cifar10"

    X_train, Y_train, X_test, Y_test, label_map = load_cifar10(cifar10_path)

    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)



