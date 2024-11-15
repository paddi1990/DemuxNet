# demuxnet/__main__.py

import argparse

import torch
from demuxnet.utils import read_rds
from demuxnet.models import DNNClassifier
from demuxnet.utils import split_data_by_cmo_label,select_top_features_by_non_zero_count,convert_labels_to_int,MyDataset,accuracy_score
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split as ts
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="DemuxNet: A tool for single-cell demultiplexing using DNN-based models.")

    # Input gene expression matrix file in RDS format
    parser.add_argument('-i', '--input', type=str, required=True, 
                        help="Path to the gene expression matrix in RDS format (input file).")

    # Model type (e.g., 'DNN', you can extend this to more model types)
    parser.add_argument('-model', '--model', type=str, choices=['DNN'], default='DNN',
                        help="The machine learning model to use. Currently supports: 'DNN'. Default is 'DNN'.")

    # Number of features to select (default to 6000)
    parser.add_argument('-feature', '--features', type=int, default=6000,
                        help="Number of top features to select based on non-zero counts. Default is 6000.")

    # Output file path for saving predictions
    parser.add_argument('-out', '--output', type=str, required=True,
                        help="Path to save the predicted labels (output file, e.g., prediction.csv).")

    # Optional: Add verbosity or debug level
    parser.add_argument('-v', '--verbose', action='store_true', 
                        help="Enable verbose output for debugging and detailed logs.")

    # Optional: Add a flag for whether to perform feature scaling
    parser.add_argument('--scale', action='store_true', 
                        help="Flag to indicate whether to scale the features before model training.")
    
    return parser.parse_args()





def main_backup():

    print("DemuxNet is running!")
    args = parse_args()

    print("Reading input file!")
    raw_data=read_rds(args.input)


    train_data, test_data, train_class_list = split_data_by_cmo_label(raw_data)

    

    data_top_n, top_n_cols=select_top_features_by_non_zero_count(train_data,top_n=args.features)

    
    train_class_list_int,label_reverse_mapping = convert_labels_to_int(train_class_list)


    # Get the number of unique classes (unique labels)
    num_classes = train_class_list_int.nunique()


    dnn_classifier=DNNClassifier(input_dim=args.features, hidden_dim=100, output_dim=num_classes)
    
    #
    x_train,x_validation,y_train,y_validation = ts(data_top_n.to_numpy(),train_class_list_int.to_numpy(),test_size=0.2,random_state=0, shuffle=True)

    train_dataset=MyDataset(x_train,y_train)
    train_loader=DataLoader(train_dataset, batch_size=32, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(dnn_classifier.parameters(), lr=0.001)

    num_epochs=20
    # 训练模型
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            inputs=Variable(inputs).to(torch.float32)
            labels=Variable(labels).to(torch.long)
            # 将梯度缓存清零
            optimizer.zero_grad()

            # forward propagation, loss and back propagation
            outputs = dnn_classifier(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # 更新参数
            optimizer.step()

            # 输出统计信息
            running_loss += loss.item()
            if i % 20 == 0:
                print('Training process: epoch [%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0

    # Validation process
    print("Training completed.")
    print("Validation process:")



    validation_dataset=MyDataset(x_validation,y_validation)
    validation_loader=DataLoader(validation_dataset, batch_size=32, shuffle=False)
    validation_prediction=[]
    for i, data in enumerate(validation_loader, 0):
        inputs, labels = data

        inputs=Variable(inputs).to(torch.float32)
        labels=Variable(labels).to(torch.long)
        
        outputs = dnn_classifier(inputs)
        pred = list(torch.max(outputs, 1)[1].numpy())
        validation_prediction.extend(pred)


    accuracy = accuracy_score(y_validation,validation_prediction)

    #########
    print("Validation accuracy:\t",accuracy)
    

    #Inference
    print("Inference process:")

    test_dataset=MyDataset(test_data[top_n_cols].to_numpy(),np.array([0]*len(test_data[top_n_cols])))



    test_loader=DataLoader(test_dataset, batch_size=32, shuffle=False)
    test_prediction=[]
    for i, data in enumerate(test_loader, 0):
        inputs, labels = data

        inputs=Variable(inputs).to(torch.float32)
        labels=Variable(labels).to(torch.long)
        
        outputs = dnn_classifier(inputs)
        pred = list(torch.max(outputs, 1)[1].numpy())
        test_prediction.extend(pred)
    
    #convert int predictions to CMO labels

    test_prediction_CMO=[label_reverse_mapping[p] for p in test_prediction]


    prediction=pd.DataFrame(test_prediction_CMO,index=test_data.index,columns=["label"])
    prediction["source"]=["prediction"]*len(prediction)
    print(prediction)

    train=pd.DataFrame(train_class_list,index=data_top_n.index,columns=["label"])
    train["source"]=["annotation"]*len(train)
    print(train)
    
    merged_df=pd.concat([prediction,train],ignore_index=False).reindex(raw_data.index).to_csv(args.output)



def main():
    # Print the initial message indicating that DemuxNet is running
    print("DemuxNet is running!")

    # Parse the command-line arguments
    args = parse_args()

    # Print message about reading the input file
    print("Reading input file!")
    raw_data = read_rds(args.input)

    # Split the data based on the CMO labels (train/test split)
    train_data, test_data, train_class_list = split_data_by_cmo_label(raw_data)

    # Select top features based on non-zero counts from the training data
    data_top_n, top_n_cols = select_top_features_by_non_zero_count(train_data, top_n=args.features)

    # Convert the class labels to integer format and prepare the reverse mapping
    train_class_list_int, label_reverse_mapping = convert_labels_to_int(train_class_list)

    # Get the number of unique classes from the training labels
    num_classes = train_class_list_int.nunique()

    # Initialize the DNN classifier model with appropriate dimensions
    dnn_classifier = DNNClassifier(input_dim=args.features, hidden_dim=100, output_dim=num_classes)

    # Split the training data into training and validation sets
    x_train, x_validation, y_train, y_validation = ts(
        data_top_n.to_numpy(), 
        train_class_list_int.to_numpy(),
        test_size=0.2,
        random_state=0,
        shuffle=True
    )

    # Prepare the datasets and data loaders for training and validation
    train_dataset = MyDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Define loss function (CrossEntropyLoss) and optimizer (Adam)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(dnn_classifier.parameters(), lr=0.001)

    # Set the number of epochs for training
    num_epochs = 20

    # Start the training loop
    for epoch in range(num_epochs):
        running_loss = 0.0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            # Convert inputs and labels to appropriate data types and move to device
            inputs = Variable(inputs).to(torch.float32)
            labels = Variable(labels).to(torch.long)

            # Zero the gradients before backpropagation
            optimizer.zero_grad()

            # Perform forward pass and compute the loss
            outputs = dnn_classifier(inputs)
            loss = criterion(outputs, labels)

            # Backpropagate the gradients
            loss.backward()

            # Update the model parameters
            optimizer.step()

            # Accumulate the loss for reporting
            running_loss += loss.item()

            # Print the training loss every 20 iterations
            if i % 20 == 0:
                print(f'Training process: epoch [{epoch+1}, {i+1:5d}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0

    # Print training completion message
    print("Training completed.")

    # Validation process
    print("Validation process:")

    # Prepare validation dataset and loader
    validation_dataset = MyDataset(x_validation, y_validation)
    validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)

    # Store the predictions for validation
    validation_prediction = []
    for i, data in enumerate(validation_loader, 0):
        inputs, labels = data

        # Convert inputs and labels to appropriate data types
        inputs = Variable(inputs).to(torch.float32)
        labels = Variable(labels).to(torch.long)

        # Perform forward pass for validation data
        outputs = dnn_classifier(inputs)
        pred = list(torch.max(outputs, 1)[1].numpy())

        # Append predictions to the list
        validation_prediction.extend(pred)

    # Calculate the accuracy of the validation predictions
    accuracy = accuracy_score(y_validation, validation_prediction)

    # Print the validation accuracy
    print("Validation accuracy:", accuracy)

    # Inference process
    print("Inference process:")

    # Prepare test dataset and loader for inference
    test_dataset = MyDataset(test_data[top_n_cols].to_numpy(), np.array([0] * len(test_data[top_n_cols])))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Store the predictions for test data
    test_prediction = []
    for i, data in enumerate(test_loader, 0):
        inputs, labels = data

        # Convert inputs and labels to appropriate data types
        inputs = Variable(inputs).to(torch.float32)
        labels = Variable(labels).to(torch.long)

        # Perform forward pass for test data
        outputs = dnn_classifier(inputs)
        pred = list(torch.max(outputs, 1)[1].numpy())

        # Append predictions to the list
        test_prediction.extend(pred)

    # Convert the integer predictions to CMO class labels
    test_prediction_CMO = [label_reverse_mapping[p] for p in test_prediction]

    # Create a DataFrame to store the test predictions
    prediction = pd.DataFrame(test_prediction_CMO, index=test_data.index, columns=["label"])
    prediction["source"] = ["prediction"] * len(prediction)
    print(prediction)

    # Create a DataFrame to store the training labels
    train = pd.DataFrame(train_class_list, index=data_top_n.index, columns=["label"])
    train["source"] = ["annotation"] * len(train)
    print(train)

    # Merge the predictions and training data into a single DataFrame
    merged_df = pd.concat([prediction, train], ignore_index=False).reindex(raw_data.index)

    # Save the merged DataFrame to the output file
    merged_df.to_csv(args.output)



if __name__ == "__main__":
    main()


