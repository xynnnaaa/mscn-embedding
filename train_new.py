import argparse
import time
import os

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
        samples, predicates, joins, targets = Variable(samples), Variable(predicates), Variable(joins), Variable(
            targets)
        sample_masks, predicate_masks, join_masks = Variable(sample_masks), Variable(predicate_masks), Variable(
            join_masks)

        t = time.time()
        outputs = model(samples, predicates, joins, sample_masks, predicate_masks, join_masks)
        t_total += time.time() - t

        for i in range(outputs.data.shape[0]):
            preds.append(outputs.data[i])

    return preds, t_total


def print_qerror(preds_unnorm, labels_unnorm):
    qerror = []
    for i in range(len(preds_unnorm)):
        if float(preds_unnorm[i]) > float(labels_unnorm[i]):
            qerror.append(float(preds_unnorm[i]) / float(labels_unnorm[i]))
        else:
            qerror.append(float(labels_unnorm[i]) / float(preds_unnorm[i]))

    print("Median: {}".format(np.median(qerror)))
    print("90th percentile: {}".format(np.percentile(qerror, 90)))
    print("95th percentile: {}".format(np.percentile(qerror, 95)))
    print("99th percentile: {}".format(np.percentile(qerror, 99)))
    print("Max: {}".format(np.max(qerror)))
    print("Mean: {}".format(np.mean(qerror)))


def print_qerror_new(preds_unnorm, labels_unnorm, join_counts):
    """
    打印不同 Join 数量的 q-error 统计量。

    :param preds_unnorm: 预测值列表
    :param labels_unnorm: 真实值列表
    :param join_counts: 每个查询的 Join 数量列表
    """
    # 初始化字典来存储不同 Join 数量的 q-error 列表
    qerrors = {1: [], 2: [], 3: []}

    # 计算 q-error
    for pred, label, join_cnt in zip(preds_unnorm, labels_unnorm, join_counts):
        if pred > label:
            qerror = pred / label
        else:
            qerror = label / pred
        qerrors[join_cnt].append(qerror)

    # 打印统计量
    for join_cnt in sorted(qerrors.keys()):
        qerror_list = qerrors[join_cnt]
        if qerror_list:
            print(f"Join Count: {join_cnt}")
            print("Median: {}".format(np.median(qerror_list)))
            print("90th percentile: {}".format(np.percentile(qerror_list, 90)))
            print("95th percentile: {}".format(np.percentile(qerror_list, 95)))
            print("99th percentile: {}".format(np.percentile(qerror_list, 99)))
            print("Max: {}".format(np.max(qerror_list)))
            print("Mean: {}".format(np.mean(qerror_list)))
            print("")

    # 打印所有查询的 q-error 统计量
    all_qerrors = [qerror for sublist in qerrors.values() for qerror in sublist]
    print("All Queries Q-Error Statistics:")
    print("Median: {}".format(np.median(all_qerrors)))
    print("90th percentile: {}".format(np.percentile(all_qerrors, 90)))
    print("95th percentile: {}".format(np.percentile(all_qerrors, 95)))
    print("99th percentile: {}".format(np.percentile(all_qerrors, 99)))
    print("Max: {}".format(np.max(all_qerrors)))
    print("Mean: {}".format(np.mean(all_qerrors)))


def train_and_predict(workload_name, train_set, num_queries, num_epochs, batch_size, hid_units, cuda, 
                      embedding_path_train, embedding_path_test):
    # Load training and validation data
    embedding_dim = 768
    dicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, max_num_joins, max_num_predicates, train_data, test_data = get_train_datasets(
        train_set, num_queries, embedding_path_train)
    table2vec, column2vec, op2vec, join2vec = dicts

    # Train model
    sample_feats = len(table2vec) + embedding_dim
    predicate_feats = len(column2vec) + len(op2vec) + 1
    join_feats = len(join2vec)

    model = SetConv(sample_feats, predicate_feats, join_feats, hid_units)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if cuda:
        model.cuda()

    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    # model.train()

    # 初始化最佳验证损失和模型路径
    best_val_loss = float('inf')
    model_save_dir = "saved_models/"
    os.makedirs(model_save_dir, exist_ok=True)
    best_model_path = os.path.join(model_save_dir, f"{train_set}_best_model.pth")

    for epoch in range(num_epochs):
        loss_total = 0.

        # 训练阶段
        model.train()
        for batch_idx, data_batch in enumerate(train_data_loader):
            samples, predicates, joins, targets, sample_masks, predicate_masks, join_masks = data_batch

            if cuda:
                samples, predicates, joins, targets = samples.cuda(), predicates.cuda(), joins.cuda(), targets.cuda()
                sample_masks, predicate_masks, join_masks = sample_masks.cuda(), predicate_masks.cuda(), join_masks.cuda()
            samples, predicates, joins, targets = Variable(samples), Variable(predicates), Variable(joins), Variable(targets)
            sample_masks, predicate_masks, join_masks = Variable(sample_masks), Variable(predicate_masks), Variable(join_masks)

            optimizer.zero_grad()
            outputs = model(samples, predicates, joins, sample_masks, predicate_masks, join_masks)
            loss = qerror_loss(outputs, targets.float(), min_val, max_val)
            loss_total += loss.item()
            loss.backward()
            optimizer.step()

        print("Epoch {}, Train Loss: {}".format(epoch, loss_total / len(train_data_loader)))

        # 验证阶段
        model.eval()
        val_loss_total = 0.
        with torch.no_grad():
            for batch_idx, data_batch in enumerate(test_data_loader):
                samples, predicates, joins, targets, sample_masks, predicate_masks, join_masks = data_batch

                if cuda:
                    samples, predicates, joins, targets = samples.cuda(), predicates.cuda(), joins.cuda(), targets.cuda()
                    sample_masks, predicate_masks, join_masks = sample_masks.cuda(), predicate_masks.cuda(), join_masks.cuda()
                samples, predicates, joins, targets = Variable(samples), Variable(predicates), Variable(joins), Variable(targets)
                sample_masks, predicate_masks, join_masks = Variable(sample_masks), Variable(predicate_masks), Variable(join_masks)

                outputs = model(samples, predicates, joins, sample_masks, predicate_masks, join_masks)
                val_loss = qerror_loss(outputs, targets.float(), min_val, max_val)
                val_loss_total += val_loss.item()

        val_loss_avg = val_loss_total / len(test_data_loader)
        print("Epoch {}, Validation Loss: {}".format(epoch, val_loss_avg))

        # 如果当前验证损失更低，则保存模型
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            torch.save(model.state_dict(), best_model_path)
            print("Saved new best model with validation loss: {:.4f}".format(val_loss_avg))

    # 加载最佳模型
    print("Loading best model with validation loss: {:.4f}".format(best_val_loss))
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    
    # Get final training and validation set predictions
    preds_train, t_total = predict(model, train_data_loader, cuda)
    print("Prediction time per training sample: {}".format(t_total / len(labels_train) * 1000))

    preds_test, t_total = predict(model, test_data_loader, cuda)
    print("Prediction time per validation sample: {}".format(t_total / len(labels_test) * 1000))

    # Unnormalize
    preds_train_unnorm = unnormalize_labels(preds_train, min_val, max_val)
    labels_train_unnorm = unnormalize_labels(labels_train, min_val, max_val)

    preds_test_unnorm = unnormalize_labels(preds_test, min_val, max_val)
    labels_test_unnorm = unnormalize_labels(labels_test, min_val, max_val)
    
    # 从 train_join_cnt.txt 文件中读取 Join 数量
    # join_counts = []
    # with open("train_join_cnt.txt", 'r') as f:
    #     for line in f:
    #         join_counts.append(int(line.strip()))
    # num_train = int(len(join_counts) * 0.9)
    # join_counts_train = join_counts[:num_train]
    # join_counts_test = join_counts[num_train:]

    # Print metrics
    print("\nQ-Error training set:")
    print_qerror(preds_train_unnorm, labels_train_unnorm)
    # print_qerror_new(preds_train_unnorm, labels_train_unnorm, join_counts_train)

    print("\nQ-Error validation set:")
    print_qerror(preds_test_unnorm, labels_test_unnorm)
    # print_qerror_new(preds_test_unnorm, labels_test_unnorm, join_counts_test)
    print("")


    # # Load test data
    # file_name = "workloads/" + workload_name
    # joins, predicates, tables, label = load_data(file_name)
    # num_tables_per_query_list = [len(query) for query in tables]
    # embeddings = load_query_specific_table_embeddings(embedding_path_test, len(tables), num_tables_per_query_list)

    # # Get feature encoding and proper normalization
    # samples_test = encode_samples(tables, embeddings, table2vec)
    # predicates_test, joins_test = encode_data(predicates, joins, column_min_max_vals, column2vec, op2vec, join2vec)
    # labels_test, _, _ = normalize_labels(label, min_val, max_val)

    # print("Number of test samples: {}".format(len(labels_test)))

    # max_num_predicates = max([len(p) for p in predicates_test])
    # max_num_joins = max([len(j) for j in joins_test])

    # # Get test set predictions
    # test_data = make_dataset(samples_test, predicates_test, joins_test, labels_test, max_num_joins, max_num_predicates)
    # test_data_loader = DataLoader(test_data, batch_size=batch_size)

    # preds_test, t_total = predict(model, test_data_loader, cuda)
    # print("Prediction time per test sample: {}".format(t_total / len(labels_test) * 1000))

    # # Unnormalize
    # preds_test_unnorm = unnormalize_labels(preds_test, min_val, max_val)

    # # Print metrics
    # print("\nQ-Error " + workload_name + ":")
    # print_qerror(preds_test_unnorm, label)

    # # Write predictions
    # file_name = "results/predictions_" + workload_name + ".csv"
    # os.makedirs(os.path.dirname(file_name), exist_ok=True)
    # with open(file_name, "w") as f:
    #     for i in range(len(preds_test_unnorm)):
    #         f.write(str(preds_test_unnorm[i]) + "," + label[i] + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("testset", help="synthetic, scale, or job-light")
    parser.add_argument("--trainset")
    parser.add_argument("--queries", help="number of training queries (default: 10000)", type=int, default=10000)
    parser.add_argument("--epochs", help="number of epochs (default: 10)", type=int, default=10)
    parser.add_argument("--batch", help="batch size (default: 1024)", type=int, default=1024)
    parser.add_argument("--hid", help="number of hidden units (default: 256)", type=int, default=256)
    parser.add_argument("--cuda", help="use CUDA", action="store_true")
    parser.add_argument("--embedding_path_train", help="Path to training embeddings", default="data/embeddings/train.pt")
    parser.add_argument("--embedding_path_test", help="Path to test embeddings", default="data/embeddings/test.pt")
    args = parser.parse_args()
    train_and_predict(args.testset, args.trainset, args.queries, args.epochs, args.batch, args.hid, args.cuda, 
                      args.embedding_path_train, args.embedding_path_test)


if __name__ == "__main__":
    main()