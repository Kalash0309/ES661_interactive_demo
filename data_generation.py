import numpy as np
from sklearn.datasets import make_blobs, make_moons, make_circles
from matplotlib import pyplot
from pandas import DataFrame
import torch
import torch.utils.data as data_utils 

class DataGenerator:
    """
    Generates different datasets for classification.
    """

    def __init__(self, num_samples: int, noise: float):
        """
        :param num_samples: Number of samples to generate.
        :param noise: Noise to add to the data.
        """
        self.num_samples = num_samples
        self.noise = noise

    def generate_gauss_data(self):
        """
        Generates two gaussians with positive and negative examples.
        :return: Data points and labels.
        """

        # generate 2d classification dataset
        X, y = make_blobs(n_samples=self.num_samples, centers=2, n_features=2, random_state=1)
        X_train = torch.from_numpy(X).float()
        y_train = torch.from_numpy(y).float()
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
        X_test = torch.linspace(-5, 5, 100).unsqueeze(1)
        return X_train, y_train, train_loader, X_test
    
    def generate_moon_data(self):
        """
        Generates two moons popularly known as the moons dataset.
        :return: Data points and labels.
        """

        # generate 2d classification dataset
        X, y = make_moons(n_samples=self.num_samples, noise=self.noise, random_state=1)
        X_train = torch.from_numpy(X).float()
        y_train = torch.from_numpy(y).float()
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
        X_test = torch.linspace(-5, 5, 100).unsqueeze(1)
        return X_train, y_train, train_loader, X_test
    
    def generate_circles_data(self):
        """
        Generates two concentric circles.
        :return: Data points and labels.
        """

        # generate 2d classification dataset
        X, y = make_circles(n_samples=self.num_samples, noise=self.noise, random_state=1)
        X_train = torch.from_numpy(X).float()
        y_train = torch.from_numpy(y).float()
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
        X_test = torch.linspace(-5, 5, 100).unsqueeze(1)
        return X_train, y_train, train_loader, X_test

    def generate_spiral_data(self):
        # Generate angles for the spirals
        theta = np.sqrt(np.random.rand(self.num_samples)) * (2 * np.pi)
        # Class 0 spiral points
        r0 = 3 * theta + np.random.randn(self.num_samples) * self.noise
        x0 = r0 * np.cos(theta)
        y0 = r0 * np.sin(theta)
        
        # Class 1 spiral points
        r1 = 3 * theta + np.random.randn(self.num_samples) * self.noise
        x1 = -r1 * np.cos(theta)
        y1 = -r1 * np.sin(theta)

        # Combine the points and labels for both classes
        class_0_points = np.column_stack((x0, y0))
        class_1_points = np.column_stack((x1, y1))
        X = np.vstack((class_0_points, class_1_points))
        y = np.hstack((np.zeros(self.num_samples), np.ones(self.num_samples)))

        return X, y
    
    def generate_dataset_1(self):
        torch.manual_seed(711)
        x = torch.linspace(-2, 2, self.num_samples)
        y = x.pow(3) - x.pow(2) + self.noise*torch.randn(x.size())
        x_train = torch.unsqueeze(x, dim=1)
        y_train = torch.unsqueeze(y, dim=1)
        train_loader = data_utils.DataLoader(
            data_utils.TensorDataset(x_train, y_train), 
            batch_size=self.num_samples
        )
        x_test = torch.linspace(-2, 2, 100).unsqueeze(-1)
        return x_train, y_train, train_loader, x_test
    
    def generate_sinusoid_data(self):
        # create simple sinusoid data set
        X_train = (torch.rand(self.num_samples) * 8).unsqueeze(-1)
        y_train = torch.sin(X_train) + torch.randn_like(X_train) * self.noise
        train_loader = data_utils.DataLoader(
            data_utils.TensorDataset(X_train, y_train), 
            batch_size=self.num_samples
        )
        X_test = torch.linspace(-5, 13, 100).unsqueeze(-1)
        return X_train, y_train, train_loader, X_test
        