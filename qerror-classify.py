import argparse
import time
import os
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from mscn.util import *
from mscn.data import get_train_datasets, load_data, make_dataset, load_query_specific_table_embeddings
from mscn.model import SetConv

def unnormalize_torch(vals, min_val, max_val):
    vals = (vals * (max_val - min_val)) + min_val
    return torch.exp(vals)

def qerror_loss(preds, targets, min_val, max_val):
    qerror = []
    preds = unnormalize_torch(preds, min_val, max_val)
    targets = unnormalize_torch(targets, min_val, max_val)

    for i in range(len(targets)):
        if (preds[i] > targets[i]).cpu().data.numpy()[0]:
            qerror.append(preds[i] / targets[i])
        else:
            qerror.append(targets[i] / preds[i])
    return torch.mean(torch.cat(qerror))

def predict(model, data_loader, cuda):
    preds = []
    t_total = 0.

    model.eval()
    for batch_idx, data_batch in enumerate(data_loader):
        samples, predicates, joins, targets, sample_masks, predicate_masks, join_masks = data_batch

        if cuda:
            samples, predicates, joins, targets = samples.cuda(), predicates.cuda(), joins.cuda(), targets.cuda()
            sample_masks, predicate_masks, join_masks = sample_masks.cuda(), predicate_masks.cuda(), join_masks.cuda()
        samples, predicates, joins, targets = Variable(samples), Variable(predicates), Variable(joins), Variable(targets)
        sample_masks, predicate_masks, join_masks = Variable(sample_masks), Variable(predicate_masks), Variable(join_masks)

        t = time.time()
        outputs = model(samples, predicates, joins, sample_masks, predicate_masks, join_masks)
        t_total += time.time() - t

        for i in range(outputs.data.shape[0]):
            preds.append(outputs.data[i])

    return preds, t_total

def print_qerror_new(preds_unnorm, labels_unnorm, join_counts):
    """
    打印不同 Join 数量的 q-error 统计量。

    :param preds_unnorm: 预测值列表
    :param labels_unnorm: 真实值列表
    :param join_counts: 每个查询的 Join 数量列表
    """
    qerrors = {0: [], 1: []}

    for pred, label, join_cnt in zip(preds_unnorm, labels_unnorm, join_counts):
        if pred > label:
            qerror = pred / label
        else:
            qerror = label / pred
        qerrors[join_cnt].append(qerror)

    for join_cnt in sorted(qerrors.keys()):
        qerror_list = qerrors[join_cnt]
        if qerror_list:
            if join_cnt == 0:
                print("采样未命中：")
            else:
                print("采样命中：")
            
            # print(f"Join Count: {join_cnt}")
            print("Median: {}".format(np.median(qerror_list)))
            print("90th percentile: {}".format(np.percentile(qerror_list, 90)))
            print("95th percentile: {}".format(np.percentile(qerror_list, 95)))
            print("99th percentile: {}".format(np.percentile(qerror_list, 99)))
            print("Max: {}".format(np.max(qerror_list)))
            print("Mean: {}".format(np.mean(qerror_list)))
            print("")

    all_qerrors = [qerror for sublist in qerrors.values() for qerror in sublist]
    print("All Queries Q-Error Statistics:")
    print("Median: {}".format(np.median(all_qerrors)))
    print("90th percentile: {}".format(np.percentile(all_qerrors, 90)))
    print("95th percentile: {}".format(np.percentile(all_qerrors, 95)))
    print("99th percentile: {}".format(np.percentile(all_qerrors, 99)))
    print("Max: {}".format(np.max(all_qerrors)))
    print("Mean: {}".format(np.mean(all_qerrors)))

def train_and_predict(workload_name, train_set, num_queries, batch_size, hid_units, cuda, 
                      embedding_path_train):
    # Load training and validation data
    embedding_dim = 768
    dicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, max_num_joins, max_num_predicates, train_data, test_data = get_train_datasets(
        train_set, num_queries, embedding_path_train)
    table2vec, column2vec, op2vec, join2vec = dicts

    # Load best model
    best_model_path = "saved_models/" + train_set + "_best_model.pth"
    print("Loading best model from: {}".format(best_model_path))

    sample_feats = len(table2vec) + embedding_dim
    predicate_feats = len(column2vec) + len(op2vec) + 1
    join_feats = len(join2vec)

    model = SetConv(sample_feats, predicate_feats, join_feats, hid_units)

    if cuda:
        model.cuda()

    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    # Create data loaders
    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    # Get predictions
    preds_train, t_total = predict(model, train_data_loader, cuda)
    print("Prediction time per training sample: {}".format(t_total / len(labels_train) * 1000))

    preds_test, t_total = predict(model, test_data_loader, cuda)
    print("Prediction time per validation sample: {}".format(t_total / len(labels_test) * 1000))

    # Unnormalize
    preds_train_unnorm = unnormalize_labels(preds_train, min_val, max_val)
    labels_train_unnorm = unnormalize_labels(labels_train, min_val, max_val)

    preds_test_unnorm = unnormalize_labels(preds_test, min_val, max_val)
    labels_test_unnorm = unnormalize_labels(labels_test, min_val, max_val)

    # Load join counts from file
    join_counts = []
    with open("train_embedding_class.txt", 'r') as f:
        for line in f:
            join_counts.append(int(line.strip()))

    # Split join counts into training and validation sets
    num_train = int(len(join_counts) * 0.9)
    join_counts_train = join_counts[:num_train]
    join_counts_test = join_counts[num_train:]

    # Print q-error statistics
    print("\nQ-Error training set:")
    print_qerror_new(preds_train_unnorm, labels_train_unnorm, join_counts_train)

    print("\nQ-Error validation set:")
    print_qerror_new(preds_test_unnorm, labels_test_unnorm, join_counts_test)
    print("")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("testset", help="synthetic, scale, or job-light")
    parser.add_argument("--trainset")
    parser.add_argument("--queries", help="number of training queries (default: 10000)", type=int, default=10000)
    parser.add_argument("--batch", help="batch size (default: 1024)", type=int, default=1024)
    parser.add_argument("--hid", help="number of hidden units (default: 256)", type=int, default=256)
    parser.add_argument("--cuda", help="use CUDA", action="store_true")
    parser.add_argument("--embedding_path_train", help="Path to training embeddings", default="data/embeddings/train.pt")
    args = parser.parse_args()
    train_and_predict(args.testset, args.trainset, args.queries, args.batch, args.hid, args.cuda, args.embedding_path_train)

if __name__ == "__main__":
    main()