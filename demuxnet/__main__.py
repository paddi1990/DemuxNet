# demuxnet/__main__.py

import argparse

import torch
from utils import read_rds
from models import DNNClassifier
from utils import split_data_by_cmo_label,select_top_features_by_non_zero_count,convert_labels_to_int,MyDataset,accuracy_score
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split as ts
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


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





def main():

    print("DemuxNet is running!")
    args = parse_args()

    print("Reading input file!")
    data=read_rds(args.input)
    print(data.head())

    train_data, test_data, train_class_list = split_data_by_cmo_label(data)

    

    data_top_n, top_n_cols=select_top_features_by_non_zero_count(train_data,top_n=args.features)

    
    train_class_list_int,label_mapping = convert_labels_to_int(train_class_list)


    # Get the number of unique classes (unique labels)
    num_classes = train_class_list_int.nunique()
    print(63,num_classes,train_class_list_int.unique())

    dnn_classifier=DNNClassifier(input_dim=args.features, hidden_dim=100, output_dim=num_classes)
    
    #
    x_train,x_valiation,y_train,y_valiation = ts(data_top_n.to_numpy(),train_class_list_int.to_numpy(),test_size=0.2,random_state=0, shuffle=True)

    train_data=MyDataset(x_train,y_train)
    train_loader=DataLoader(train_data, batch_size=32, shuffle=True)

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

    # Your main script logic here
    data=read_rds("/home/wuyou/Projects/scRNA-seq/20230506_full_matrix.rds")
    #print(data)

    valiation_data=MyDataset(x_valiation,y_valiation)
    valiation_loader=DataLoader(valiation_data, batch_size=32, shuffle=False)
    result=[]
    for i, data in enumerate(valiation_loader, 0):
        inputs, labels = data

        inputs=Variable(inputs).to(torch.float32)
        labels=Variable(labels).to(torch.long)
        
        outputs = dnn_classifier(inputs)
        pred = list(torch.max(outputs, 1)[1].numpy())
        result.extend(pred)


    accuracy = accuracy_score(y_valiation,result)

    #########
    print("Validation accuracy:\t",accuracy)
    

    print("Inference process:")



if __name__ == "__main__":
    main()


