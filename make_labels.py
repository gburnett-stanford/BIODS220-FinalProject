import os 
import argparse 
import numpy as np

def main(test_dir): 

    for i in range(5): 

        filepath = os.path.join(test_dir, str(i))
        num_files = len(os.listdir(filepath))

        if i==0:
            labels = np.repeat(i, num_files)
        else:  
            labels = np.append(labels, np.repeat(i, num_files))

    np.savetxt(os.path.join(test_dir, 'labels.csv'), labels, delimiter=",")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--test_dir', type=str, help='Directory with the test data', required=True)

    args = parser.parse_args()

    main(args.test_dir) 