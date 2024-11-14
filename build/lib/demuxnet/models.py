import torch
import torch.nn as nn





class DNNClassifier(nn.Module):
    """
    A deep neural network (DNN) classifier with multiple hidden layers.
    This model takes an input vector, processes it through several hidden layers, 
    and outputs a classification prediction.

    Args:
        input_dim (int): The number of input features.
        hidden_dim (int): The number of units in each hidden layer.
        output_dim (int): The number of output classes.
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Initialize the layers of the DNN model.

        Parameters:
            input_dim (int): The number of input features.
            hidden_dim (int): The number of units in each hidden layer.
            output_dim (int): The number of output classes.
        """
        super(DNNClassifier, self).__init__()

        # Define the layers
        self.input_layer = nn.Linear(input_dim, hidden_dim)  # Input to first hidden layer
        self.hidden_layer1 = nn.Linear(hidden_dim, hidden_dim)  # First hidden layer
        self.hidden_layer2 = nn.Linear(hidden_dim, hidden_dim)  # Second hidden layer
        self.hidden_layer3 = nn.Linear(hidden_dim, hidden_dim)  # Third hidden layer
        self.hidden_layer4 = nn.Linear(hidden_dim, hidden_dim)  # Fourth hidden layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)  # Output layer (classification)

        # Activation functions
        self.relu = nn.ReLU()  # ReLU activation function
        self.leaky_relu = nn.LeakyReLU()  # Leaky ReLU activation function
        self.dropout = nn.Dropout(p=0.1)  # Dropout layer for regularization

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output prediction after passing through the network.
        """
        # Pass through the input layer and apply activation
        x = self.input_layer(x)
        x = self.leaky_relu(x)  # Apply LeakyReLU after input layer
        
        # Pass through hidden layers with activations and optional dropout
        x = self.hidden_layer1(x)
        x = self.leaky_relu(x)  # Apply LeakyReLU after first hidden layer
        # x = self.dropout(x)  # Optional: Uncomment for dropout after first hidden layer

        x = self.hidden_layer2(x)
        x = self.leaky_relu(x)  # Apply LeakyReLU after second hidden layer
        # x = self.dropout(x)  # Optional: Uncomment for dropout after second hidden layer
        
        x = self.hidden_layer3(x)
        x = self.leaky_relu(x)  # Apply LeakyReLU after third hidden layer
        # x = self.dropout(x)  # Optional: Uncomment for dropout after third hidden layer
        
        x = self.hidden_layer4(x)
        x = self.leaky_relu(x)  # Apply LeakyReLU after fourth hidden layer
        # x = self.dropout(x)  # Optional: Uncomment for dropout after fourth hidden layer

        # Final output layer (no activation here as it's for classification)
        x = self.output_layer(x)
        return x  # Output logits or prediction scores



