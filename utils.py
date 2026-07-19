import logging
import os
import pickle
import time
from collections import defaultdict, deque

import numpy as np


def get_logger(dataset):
    os.makedirs("./log", exist_ok=True)
    pathname = "./log/{}_{}.txt".format(dataset, time.strftime("%m-%d_%H-%M-%S"))
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s",
                                  datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(pathname)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def format_table(headers, rows):
    """Tạo bảng text đơn giản để tránh phụ thuộc prettytable trên Kaggle.

    Args:
        headers: Danh sách tiêu đề cột.
        rows: Danh sách các dòng dữ liệu.

    Returns:
        Chuỗi bảng đã căn cột.
    """
    table = [[str(cell) for cell in headers]]
    table.extend([[str(cell) for cell in row] for row in rows])
    widths = [max(len(row[i]) for row in table) for i in range(len(headers))]
    separator = "+{}+".format("+".join("-" * (width + 2) for width in widths))

    lines = [separator]
    for row_index, row in enumerate(table):
        line = "|{}|".format("|".join(" {} ".format(cell.ljust(widths[i]))
                                      for i, cell in enumerate(row)))
        lines.append(line)
        if row_index == 0:
            lines.append(separator)
    lines.append(separator)
    return "\n".join(lines)


def classification_metrics(y_true, y_pred, average="macro"):
    """Tính precision/recall/F1 cơ bản mà không cần scikit-learn.

    Args:
        y_true: Nhãn thật.
        y_pred: Nhãn dự đoán.
        average: "macro" để lấy trung bình, hoặc None để trả F1 từng nhãn.

    Returns:
        Với average="macro": tuple (precision, recall, f1, support).
        Với average=None: mảng F1 theo từng nhãn.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))

    precisions = []
    recalls = []
    f1_scores = []
    supports = []
    for label in labels:
        true_mask = y_true == label
        pred_mask = y_pred == label
        tp = np.logical_and(true_mask, pred_mask).sum()
        fp = np.logical_and(~true_mask, pred_mask).sum()
        fn = np.logical_and(true_mask, ~pred_mask).sum()

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        supports.append(true_mask.sum())

    if average is None:
        return np.asarray(f1_scores)
    if average != "macro":
        raise ValueError("Only average='macro' or average=None is supported")
    return np.mean(precisions), np.mean(recalls), np.mean(f1_scores), np.asarray(supports)


def save_file(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_file(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def convert_index_to_text(index, type):
    text = "-".join([str(i) for i in index])
    text = text + "-#-{}".format(type)
    return text


def convert_text_to_index(text):
    index, type = text.split("-#-")
    index = [int(x) for x in index.split("-")]
    return index, int(type)


def decode(outputs, entities, length):
    class Node:
        def __init__(self):
            self.THW = []                # [(tail, type)]
            self.NNW = defaultdict(set)   # {(head,tail): {next_index}}

    ent_r, ent_p, ent_c = 0, 0, 0
    decode_entities = []
    q = deque()
    for instance, ent_set, l in zip(outputs, entities, length):
        predicts = []
        nodes = [Node() for _ in range(l)]
        for cur in reversed(range(l)):
            heads = []
            for pre in range(cur+1):
                # THW
                if instance[cur, pre] > 1: 
                    nodes[pre].THW.append((cur, instance[cur, pre]))
                    heads.append(pre)
                # NNW
                if pre < cur and instance[pre, cur] == 1:
                    # cur node
                    for head in heads:
                        nodes[pre].NNW[(head,cur)].add(cur)
                    # post nodes
                    for head,tail in nodes[cur].NNW.keys():
                        if tail >= cur and head <= pre:
                            nodes[pre].NNW[(head,tail)].add(cur)
            # entity
            for tail,type_id in nodes[cur].THW:
                if cur == tail:
                    predicts.append(([cur], type_id))
                    continue
                q.clear()
                q.append([cur])
                while len(q) > 0:
                    chains = q.pop()
                    for idx in nodes[chains[-1]].NNW[(cur,tail)]:
                        if idx == tail:
                            predicts.append((chains + [idx], type_id))
                        else:
                            q.append(chains + [idx])
        
        predicts = set([convert_index_to_text(x[0], x[1]) for x in predicts])
        decode_entities.append([convert_text_to_index(x) for x in predicts])
        ent_r += len(ent_set)
        ent_p += len(predicts)
        ent_c += len(predicts.intersection(ent_set))
    return ent_c, ent_p, ent_r, decode_entities


def cal_f1(c, p, r):
    if r == 0 or p == 0:
        return 0, 0, 0

    r = c / r if r else 0
    p = c / p if p else 0

    if r and p:
        return 2 * p * r / (p + r), p, r
    return 0, p, r
