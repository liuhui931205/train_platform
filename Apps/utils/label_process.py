# -*-coding:utf-8-*-
import os


def rm_file(dir_path):
    d = []
    path = "/data/deeplearning/dataset/training/data/released/"
    os.path.join(path, dir_path)
    li = os.listdir(path)
    for i in li:
        l_path = os.path.join(path, i)
        ji = os.listdir(l_path)
        for j in ji:
            if j.endswith(".jpg.csv"):
                d.append(os.path.join(l_path, j))
            elif j.endswith("_lane.png"):
                d.append(os.path.join(l_path, j))
            elif j == "Thumbs.db":
                d.append(os.path.join(l_path, j))
            elif j.endswith("_00_004.csv"):
                d.append(os.path.join(l_path, j))
            elif j.endswith(".json"):
                d.append(os.path.join(l_path, j))
            elif j.startswith("enhance"):
                d.append(os.path.join(l_path, j))
    for q in d:
        os.remove(q)
