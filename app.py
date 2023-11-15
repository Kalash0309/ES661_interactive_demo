import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import torch
from laplace import Laplace
from data_generation import DataGenerator
import torch.utils.data as data_utils 
from scipy.interpolate import griddata


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD


def mlp_regression(architecture_info):

    modules = []
    modules.append(torch.nn.Linear(1, architecture_info[0]['neurons']))
    # modules.append(torch.nn.ReLU())
    for i in range(len(architecture_info)-1):
        modules.append(torch.nn.ReLU())
        modules.append(torch.nn.Linear(architecture_info[i]['neurons'], architecture_info[i+1]['neurons']))
    modules.append(torch.nn.ReLU())
    modules.append(torch.nn.Linear(architecture_info[-1]['neurons'], 1))
    
    return torch.nn.Sequential(*modules)

def mlp_classification(architecture_info):

    modules = []
    modules.append(torch.nn.Linear(2, architecture_info[0]['neurons']))
    # modules.append(torch.nn.ReLU())
    for i in range(len(architecture_info)-1):
        modules.append(torch.nn.ReLU())
        modules.append(torch.nn.Linear(architecture_info[i]['neurons'], architecture_info[i+1]['neurons']))
    modules.append(torch.nn.ReLU())
    modules.append(torch.nn.Linear(architecture_info[-1]['neurons'], 1))
    # modules.append(torch.nn.Sigmoid())

    return torch.nn.Sequential(*modules)

def plot_regression(X_train, y_train, X_test, f_test, y_std):
    # fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharey=True)
    fig = plt.figure()
    plt.title('Mean and Uncertainity Predictions')
    plt.scatter(X_train.flatten(), y_train.flatten(), alpha=0.3, color='tab:blue', label='Training Data')
    plt.plot(X_test, f_test, color='black', label='Mean')
    plt.fill_between(X_test, f_test-y_std*2, f_test+y_std*2, 
                     alpha=0.3, color='tab:orange', label='95% CI')
    plt.legend()
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.tight_layout()
    plt.xlim([X_test.min(), X_test.max()])
    plt.tight_layout()
    return fig

def plot_classification(X_train, y_train, X_test, logits, var):
    probs = torch.sigmoid(torch.tensor(logits))
    probs = probs.reshape(X_test.shape[0])
    var = var.reshape(X_test.shape[0])

    # Extract x and y coordinates from the X_test tensor
    x = X_test[:, 0].numpy()
    y = X_test[:, 1].numpy()
    x_grid, y_grid = np.meshgrid(np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100))
    probs_grid = griddata((x, y), probs.numpy(), (x_grid, y_grid), method='linear')
    var_grid = griddata((x, y), var.numpy(), (x_grid, y_grid), method='linear')

    fig1 = plt.figure()
    plt.scatter(X_train[:,0], X_train[:,1], s=10, c=y_train, cmap='bwr', label='Training Data')
    plt.contourf(x_grid, y_grid, probs_grid, cmap='bwr', alpha=0.3, label='Mean')
    plt.legend()
    plt.title('Mean Predictions')
    plt.colorbar()
    plt.xlabel('$x1$')
    plt.ylabel('$x2$')
    plt.tight_layout()

    fig2 = plt.figure()
    plt.title('Uncertainity Predictions')
    plt.scatter(X_train[:,0], X_train[:,1], s=10, c=y_train, cmap='bwr', label='Training Data')
    plt.contourf(x_grid, y_grid, var_grid, cmap='bwr', alpha=0.3, label='Standard Deviation')
    plt.legend()
    plt.colorbar()
    plt.xlabel('$x1$')
    plt.ylabel('$x2$')
    plt.tight_layout()

    return fig1, fig2

def train_model(task, architecture_info, X_train, y_train, train_loader, X_test, learning_rate=0.01):
    if task=="Regression":
        model = mlp_regression(architecture_info)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for epoch in range(1000):
            for x,y in train_loader:
                optimizer.zero_grad()
                y_pred = model(x)
                # y = y.unsqueeze(0).T
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss {loss.item()}')

        la_regression = Laplace(model, 'regression', subset_of_weights='last_layer', hessian_structure='diag')
        la_regression.fit(train_loader)

        x = X_test.flatten().cpu().numpy()
        f_mu, f_var = la_regression(X_test)

        f_mu = f_mu.squeeze().detach().cpu().numpy()
        f_sigma = f_var.squeeze().detach().sqrt().cpu().numpy()
        pred_std = np.sqrt(f_sigma**2 + la_regression.sigma_noise.item()**2)

        st.pyplot(plot_regression(X_train, y_train, x, f_mu, pred_std))
        
    else:
        model = mlp_classification(architecture_info)
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for epoch in range(1000):
            for x,y in train_loader:
                optimizer.zero_grad()
                y_pred = torch.sigmoid(model(x))
                y = torch.unsqueeze(y, 1)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss {loss.item()}')

        la_classification = Laplace(model, 'regression', subset_of_weights='last_layer', hessian_structure='diag')

        la_classification.fit(train_loader)
        logits, var = la_classification(X_test)

        fig1, fig2 = plot_classification(X_train, y_train, X_test, logits, var)
        st.pyplot(fig1)
        st.pyplot(fig2)
    
def generate_data(dataset, n_samples=150, noise=0.1):
    data_generator = DataGenerator(n_samples, noise)

    if dataset == 'Dataset 1':
        return data_generator.generate_dataset_1()
    elif dataset == 'Dataset 2':
        return data_generator.generate_sinusoid_data()
    elif dataset == 'Dataset 3':
        return data_generator.generate_dataset_3()
    elif dataset == 'Dataset 4':
        return data_generator.generate_dataset_4()
    elif dataset == 'Gaussian':
        return data_generator.generate_gauss_data()
    elif dataset == 'Moons':
        return data_generator.generate_moon_data()
    elif dataset == 'Circles':
        return data_generator.generate_circles_data()
    elif dataset == 'Spiral':
        return data_generator.generate_spiral_data()
        
def main():
    st.title('End-Layer Bayesian MLP')

    # Create a sidebar
    st.sidebar.title('Adjust Hyperparameters')

    # Plus and minus buttons for Hidden Layers in the sidebar
    num_hidden_layers = st.sidebar.number_input('Number of Hidden Layers', min_value=1, step=1, value=1)

    # Dynamic creation of hidden layers and their neurons in the sidebar
    architecture_info = []
    for i in range(num_hidden_layers):
        layer_neurons = st.sidebar.number_input(f'Number of neurons in Hidden Layer {i+1}', min_value=1, step=1, value=10)
        architecture_info.append({'neurons': layer_neurons, 'activation': 'relu'})

    # Choose Task
    task = st.sidebar.selectbox('Task', ['Regression','Classification'])

    # input noise
    input_noise = st.sidebar.slider('Input Noise', min_value=0.0, max_value=1.0, step=0.01, value=0.1)

    # Sliders for Learning Rate and Epochs in the sidebar
    learning_rate = st.sidebar.slider('Learning Rate', min_value=0.001, max_value=1.0, step=0.01, value=0.01)
    epochs = 1000

    # input number of samples
    num_samples = st.sidebar.slider('Number of Samples', min_value=100, max_value=10000, step=1, value=1000)

    if task=="Regression":
        dataset = st.sidebar.selectbox('Dataset', ['Dataset 1', 'Dataset 2', 'Dataset 3', 'Dataset 4'])
        X_train, y_train, train_loader, X_test = generate_data(dataset, num_samples, input_noise)

        fig1 = plt.figure()
        plt.title(dataset)
        plt.scatter(X_train, y_train, s=10, label='Training Data')
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.legend()    
        st.pyplot(fig1)

    else:
        dataset = st.sidebar.selectbox('Dataset', ['Gaussian', 'Moons', 'Circles', 'Spiral'])
        X_train, y_train, train_loader, X_test = generate_data(dataset, num_samples, input_noise)

        fig2 = plt.figure()
        plt.title(dataset)
        plt.scatter(X_train[:,0], X_train[:,1], s=10, c=y_train, cmap='bwr', label='Training Data')
        plt.xlabel('$x1$')
        plt.ylabel('$x2$')
        plt.legend()
        st.pyplot(fig2)

    if st.button('Run Neural Network'):
        with st.spinner(text="Training in progress..."):
            train_model(task, architecture_info, X_train, y_train, train_loader, X_test, learning_rate=0.01)
        
        
if __name__ == '__main__':
    main()