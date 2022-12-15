import os 
import argparse 
import numpy as np
from sklearn.metrics import confusion_matrix

def main(test_dir, model_path): 

    # Load the predictions
    y_true = np.loadtxt(os.path.join(test_dir, 'labels.csv'), delimiter=',')
    y_pred = np.loadtxt(os.path.join(model_path, 'predictions.csv'), delimiter=',')

    # Calculate confusion matrix
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--test_dir', type=str, help='Path to test dir', required=True)
    parser.add_argument('--model_path', type=str, help='Path to the model', required=True)

    args = parser.parse_args()

    main(args.test_dir, args.model_path) 