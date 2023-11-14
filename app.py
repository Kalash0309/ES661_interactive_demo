import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import torch
from laplace import Laplace
from data_generation import DataGenerator
import torch.utils.data as data_utils 


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD



def mlp_classification(architecture_info):

    modules = []
    modules.append(torch.nn.Linear(1, architecture_info[0]['neurons']))
    modules.append(torch.nn.architecture_info[0]['activation']())
    for i in range(len(architecture_info)-1):
        modules.append(torch.nn.Linear(architecture_info[i]['neurons'], architecture_info[i+1]['neurons']))
        modules.append(torch.nn.architecture_info[i+1]['activation']())
    modules.append(torch.nn.Sigmoid())

    return torch.nn.Sequential(*modules)
    # model = Sequential()
    # model.add(Dense(2, activation='relu'))
    # for i in range(len(architecture_info)):
    #     model.add(Dense(architecture_info[i]['neurons'], activation=architecture_info[i]['activation']))
    # model.add(Dense(1, activation='sigmoid'))
    # opt = SGD(lr=learning_rate)
    # model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    # return model

def mlp_regression(architecture_info):

    modules = []
    # modules.append(torch.nn.Linear(1, architecture_info[0]['neurons']))
    # modules.append(torch.nn.ReLU())
    for i in range(len(architecture_info)):
        if i==0:
            modules.append(torch.nn.Linear(1, architecture_info[i]['neurons']))
        else:
            modules.append(torch.nn.ReLU())
            modules.append(torch.nn.Linear(architecture_info[i-1]['neurons'], architecture_info[i]['neurons']))

    return torch.nn.Sequential(*modules)


    # model = Sequential()
    # model.add(Dense(1, activation='relu'))
    # for i in range(len(architecture_info)):
    #     model.add(Dense(architecture_info[i]['neurons'], activation=architecture_info[i]['activation']))
    # opt = SGD(lr=learning_rate)
    # model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
    # return model

def train_model(task, architecture_info, X_train, y_train, train_loader, X_test, learning_rate=0.01):
    # convert torch tensors to numpy arrays
    # X_train = X_train.numpy()
    # y_train = y_train.numpy()
    # X_test = X_test.numpy()

    # model.fit(X_train, y_train, epochs=1000, batch_size=32)
    if task=="Regression":
        model = mlp_regression(architecture_info)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for epoch in range(1000):
            for x,y in train_loader:
                optimizer.zero_grad()
                y_pred = model(x.unsqueeze(0).T)
                y = y.unsqueeze(0).T
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss {loss.item()}')

        la_regression = Laplace(model, 'regression', subset_of_weights='last_layer', hessian_structure='diag')
        la_regression.fit(train_loader)
        X_test = X_test.unsqueeze(0).T
        f_mu, f_var = la_regression(X_test)

        f_mu = f_mu.squeeze().detach().cpu().numpy()
        f_sigma = f_var.squeeze().detach().sqrt().cpu().numpy()
        pred_std = np.sqrt(f_sigma**2 + la_regression.sigma_noise.item()**2)

        st.pyplot(plot_regression(X_train, y_train, x, f_mu, pred_std, 
                file_name='regression_example', plot=False))
        
    


def plot_regression(X_train, y_train, X_test, f_test, y_std, plot=True, 
                    file_name='regression_example'):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True,
                                figsize=(4.5, 2.8))
    ax1.set_title('MAP')
    ax1.scatter(X_train.flatten(), y_train.flatten(), alpha=0.3, color='tab:orange')
    ax1.plot(X_test, f_test, color='black', label='$f_{MAP}$')
    ax1.legend()

    ax2.set_title('LA')
    ax2.scatter(X_train.flatten(), y_train.flatten(), alpha=0.3, color='tab:orange')
    ax2.plot(X_test, f_test, label='$\mathbb{E}[f]$')
    ax2.fill_between(X_test, f_test-y_std*2, f_test+y_std*2, 
                     alpha=0.3, color='tab:blue', label='$2\sqrt{\mathbb{V}\,[y]}$')
    ax2.legend()
    ax1.set_ylim([-4, 6])
    ax1.set_xlim([X_test.min(), X_test.max()])
    ax2.set_xlim([X_test.min(), X_test.max()])
    ax1.set_ylabel('$y$')
    ax1.set_xlabel('$x$')
    ax2.set_xlabel('$x$')
    plt.tight_layout()
    # plt.show()
    return fig
    # if plot:
    #     plt.show()
    # else:
    #     plt.savefig(f'docs/{file_name}.png')


# Main Streamlit app
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
    task = st.sidebar.selectbox('Task', ['Classification', 'Regression'])

    # input noise
    input_noise = st.sidebar.slider('Input Noise', min_value=0.0, max_value=1.0, step=0.01, value=0.1)

    # Sliders for Learning Rate and Epochs in the sidebar
    learning_rate = st.sidebar.slider('Learning Rate', min_value=0.001, max_value=1.0, step=0.01, value=0.01)
    epochs = 1000

    # input number of samples
    num_samples = st.sidebar.slider('Number of Samples', min_value=100, max_value=10000, step=1, value=1000)

    if task=="Classification":
        dataset = st.sidebar.selectbox('Dataset', ['Gaussian', 'Moons', 'Circles', 'Spiral'])
        # Get the data points and labels
        if dataset == 'Gaussian':
            x,y = DataGenerator(num_samples, input_noise).generate_gauss_data()
        elif dataset == 'Moons':
            x,y = DataGenerator(num_samples, input_noise).generate_moon_data()
        elif dataset == 'Spiral':
            x,y = DataGenerator(num_samples, input_noise).generate_spiral_data()
        else:
            x,y = DataGenerator(num_samples, input_noise).generate_circles_data()

        X_train = torch.from_numpy(x).float()
        # X_train = X_train.unsqueeze(1)
        X_train = X_train.T
        y_train = torch.from_numpy(y).float()
        # y_train = y_train.unsqueeze(1)
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

        X_test = torch.linspace(-5, 5, 100).unsqueeze(1)


        fig1 = plt.figure(figsize=(8, 6))
        plt.scatter(X_train, y_train, s=10, label='train')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title('Training Data')
        plt.legend()
        plt.grid(True)
        st.pyplot(fig1)

        # model = mlp_classification(architecture_info)


    else:
        dataset = st.sidebar.selectbox('Dataset', ['Dataset 1', 'Dataset 2', 'Dataset 3', 'Dataset 4'])

        # Get the data points and labels
        # if dataset == 'Dataset 1':
        #     x,y = DataGenerator(num_samples, input_noise).generate_dataset_1()
        # elif dataset == 'Dataset 2':
        #     x,y = DataGenerator(num_samples, input_noise).generate_moon_data()
        # elif dataset == 'Dataset 3':
        #     x,y = DataGenerator(num_samples, input_noise).generate_spiral_data()
        # else:
        #     x,y = DataGenerator(num_samples, input_noise).generate_circles_data()

        # X_train = torch.from_numpy(x).float()
        # # X_train = X_train.T
        # y_train = torch.from_numpy(y).float()

        # train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

        # X_test = torch.linspace(-5, 5, 100).unsqueeze(1)

        x = torch.linspace(-2, 2, num_samples)
        y = x.pow(3) - x.pow(2) + input_noise*torch.randn(x.size())
        X_train = torch.unsqueeze(x, dim=1)
        y_train = torch.unsqueeze(y, dim=1)
        train_loader = data_utils.DataLoader(
            data_utils.TensorDataset(X_train, y_train), 
            batch_size=num_samples
        )
        X_test = torch.linspace(-2, 2, 100).unsqueeze(-1)

        fig1 = plt.figure(figsize=(8, 6))
        plt.scatter(X_train, y_train, s=10, label='train')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Training Data')
        plt.legend()
        plt.grid(True)
        st.pyplot(fig1)

        # model = mlp_regression(architecture_info)
    

    if st.button('Run Neural Network'):

        train_model(task, architecture_info, X_train, y_train, train_loader, X_test, learning_rate=learning_rate)


if __name__ == '__main__':
    main()
