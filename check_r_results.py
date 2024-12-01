import argparse
import logging
import pandas as pd
import numpy as np

from sklearn.metrics import multilabel_confusion_matrix, hamming_loss, precision_score,\
      recall_score, f1_score

def main():
    # set up arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb', type=str, default="bert",
                        help="Which embedding to use. The default is BERT")
    parser.add_argument('--model', type=str, default="keyatm",
                        help="Model used in R")
    opt = parser.parse_args()

    # set up logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=f"Data/pain_study/logs/output_{opt.model}_{opt.emb}.log", filemode='w', level=logging.DEBUG)

    # load results
    file_path = f"R_results/{opt.model}/pred_labels_{opt.emb}.csv"
    results = pd.read_csv(file_path)

    # subset test labels
    test_path = "Data/pain_study/test_data.csv"
    test_data = pd.read_csv(test_path)
    test_index = test_data["Index"].tolist()

    lab_cols = [col for col in test_data.columns if col.startswith("label_")]
    lab_cols = sorted(lab_cols)
    test_labels = test_data[lab_cols].to_numpy()
    r_label = results[results["Index"].isin(test_index)].loc[:, results.columns != 'Index'].to_numpy()

    logger.info(f"Hamming loss: {hamming_loss(test_labels, r_label)}")
    logger.info(f"Precision: {precision_score(y_true=test_labels,y_pred=r_label,average='samples')}")
    logger.info(f"Recall: {recall_score(y_true=test_labels,y_pred=r_label,average='samples')}")
    logger.info(f"F1 Measure: {f1_score(y_true=test_labels,y_pred=r_label,average='samples')}")
    logger.info(f"Confusion Matrix: \n {multilabel_confusion_matrix(test_labels, r_label)}")
                
    
if __name__ == "__main__":
     main()
