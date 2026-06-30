
from pathlib import Path
import numpy as np

# ======== XX ======== || ======== XX ========
# Load MNIST
# ======== XX ======== || ======== XX ========


def load_mnist_images(file_path:Path, is_test:bool=False):

    if file_path.is_file():
        if is_test:
            B=10000
        else:
            B=60000

        with open(file_path, 'rb') as f:
            import struct
            header = f.read(16)
            magic, n_images, rows, cols = struct.unpack(">IIII", header)
            pixels_bytes = f.read()

        pixels = np.frombuffer(pixels_bytes, dtype=np.uint8)
        pixels = pixels.reshape(B, 28, 28)
        return pixels
    
    return None


def load_mnist_labels(file_path:Path):

    if file_path.is_file():

        with open(file_path, 'rb') as f:
            import struct
            header = f.read(8)
            magic, n_images = struct.unpack(">II", header)
            labels_bytes = f.read()
        labels = np.frombuffer(labels_bytes, dtype=np.uint8)
        return labels
    
    return None



def load_mnist(folder_path):
    if folder_path.is_dir():

        train_img_file_path = folder_path / "train-images.idx3-ubyte"
        train_label_file_path = folder_path / "train-labels.idx1-ubyte"
        test_img_file_path = folder_path / "t10k-images.idx3-ubyte"
        test_label_file_path = folder_path / "t10k-labels.idx1-ubyte"

        # Train Data
        Xtr = load_mnist_images(train_img_file_path)
        Ytr = load_mnist_labels(train_label_file_path)

        # Test Data
        Xts = load_mnist_images(test_img_file_path, is_test=True)
        Yts = load_mnist_labels(test_label_file_path)

    return Xtr, Ytr, Xts, Yts



if __name__ == "__main__":

    cv_data_root_path = Path("/Users/irabandutta/Developer/2026-06-deep-learning-workbench/00_foundations/computer_vision/data")
    mnist_path = cv_data_root_path / "mnist"

    X_train, Y_train, X_test, Y_test = load_mnist(mnist_path)

    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)



