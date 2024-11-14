# demuxnet/__main__.py

import argparser

from utils import read_rds





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

    args = parse_args()

    print("Reading input file!")
    data=read_rds(args.input)
    print(data.head())
    


    print("DemuxNet is running!")

    # Your main script logic here
    data=read_rds("/home/wuyou/Projects/scRNA-seq/20230506_full_matrix.rds")
    #print(data)

    

if __name__ == "__main__":
    main()


