import argparse
import torch
import os

def check_if_embedding_is_zero(embedding_tensor: torch.Tensor | None) -> bool:
    """
    检查给定的 embedding 张量是否为 None 或者是全零 (或接近全零)向量。
    使用一个小的阈值来处理可能的浮点数精度问题，而不是严格的 all(==0)。
    """
    if embedding_tensor is None:
        return True
    if torch.all(embedding_tensor == 0):
        return True
    return False

def main():
    parser = argparse.ArgumentParser(
        description="为每个查询生成0/1标记，基于其所有表 embedding 是否有效。"
    )
    parser.add_argument(
        "--input_embedding_file",
        type=str,
        required=True,
        help="包含 query-specific sub-table embeddings 的 .pt 文件路径。"
    )
    parser.add_argument(
        "--output_labels_file",
        type=str,
        required=True,
        help="输出 0/1 标记的 .txt 文件路径。"
    )
    args = parser.parse_args()

    # 1. 加载之前生成的 embedding
    if not os.path.exists(args.input_embedding_file):
        print(f"错误: 输入的 embedding 文件未找到: {args.input_embedding_file}")
        exit(1)
    
    print(f"加载 embedding 文件: {args.input_embedding_file}")
    try:
        all_query_subtable_embeddings = torch.load(args.input_embedding_file)
        if not isinstance(all_query_subtable_embeddings, list):
            print(f"错误: {args.input_embedding_file} 中的内容不是预期的列表格式。")
            exit(1)
        print(f"成功加载了 {len(all_query_subtable_embeddings)} 个查询的 embedding 数据。")
    except Exception as e:
        print(f"错误: 加载 embedding 文件 {args.input_embedding_file} 失败: {e}")
        exit(1)

    query_labels = []
    num_queries_with_all_valid_embeddings = 0
    num_queries_with_at_least_one_zero_embedding = 0

    # 2. 遍历每个查询的 embedding 列表
    for query_idx, embeddings_for_one_query in enumerate(all_query_subtable_embeddings):
        if not isinstance(embeddings_for_one_query, list):
            print(f"警告: 查询 {query_idx} 的 embedding 数据不是列表格式，已跳过。实际类型: {type(embeddings_for_one_query)}")
            query_labels.append(0) # 或者标记为错误，或者不添加
            num_queries_with_at_least_one_zero_embedding += 1
            continue

        if not embeddings_for_one_query: # 如果一个查询没有任何表引用（空列表）
            print(f"警告: 查询 {query_idx} 不包含任何表 embedding，标记为 0。")
            query_labels.append(0)
            num_queries_with_at_least_one_zero_embedding += 1
            continue

        all_tables_valid = True # 假设当前查询的所有表 embedding 都有效
        for table_embedding in embeddings_for_one_query:
            if check_if_embedding_is_zero(table_embedding):
                all_tables_valid = False # 发现一个无效/全零 embedding
                break # 不需要再检查这个查询的其他表了
        
        if all_tables_valid:
            query_labels.append(1)
            num_queries_with_all_valid_embeddings += 1
        else:
            query_labels.append(0)
            num_queries_with_at_least_one_zero_embedding += 1

    print(f"\n处理完毕:")
    print(f"  所有表 embedding 均有效的查询数量: {num_queries_with_all_valid_embeddings}")
    print(f"  至少有一个表 embedding 为零/None 的查询数量: {num_queries_with_at_least_one_zero_embedding}")
    print(f"  总查询数量: {len(query_labels)}")


    # 3. 将标记写入到 .txt 文件
    try:
        output_dir = os.path.dirname(args.output_labels_file)
        if output_dir and not os.path.exists(output_dir): # 确保目录存在
            os.makedirs(output_dir)
        
        with open(args.output_labels_file, 'w') as f:
            for label in query_labels:
                f.write(str(label) + '\n')
        print(f"\n查询标记已成功写入到: {args.output_labels_file}")
    except Exception as e:
        print(f"错误: 写入标记文件 {args.output_labels_file} 失败: {e}")

if __name__ == "__main__":
    main()