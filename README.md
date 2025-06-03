```bash
# 默认采样训练命令：
nohup python3 train_new.py --trainset train --queries 50654 --embedding_path_train data/train.pt --epochs 500 test > train.log 2>&1 &

# 调整采样训练命令：
nohup python3 train_new.py --trainset train_alter --queries 50654 --embedding_path_train data/train_alter.pt --epochs 500 test > train_alter.log 2>&1 &
