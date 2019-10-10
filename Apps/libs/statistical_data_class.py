# -*-coding:utf-8-*-
import os
import cv2
import numpy as np
import multiprocessing
import time
from Apps.utils.utils import self_full_labels
import sys
import json

reload(sys)
sys.setdefaultencoding('utf-8')


class DownloadSamTask(object):

    def __init__(self, image_path, exit_flag=False):
        self.image_path = image_path
        self.exit_flag = exit_flag


class TrackSamImage(object):

    def __init__(self):
        self.li = []

    def download_image(self, download_queue, result_queue):
        while True:
            if download_queue.empty():
                time.sleep(3)
            download_task = download_queue.get()
            if not isinstance(download_task, DownloadSamTask):
                break

            if download_task.exit_flag:
                break

            image_path = download_task.image_path

            cls = int(len(self_full_labels))
            s_pixel = np.zeros(cls)
            bins = [i for i in range(cls + 1)]
            img = cv2.imread(image_path, flags=cv2.IMREAD_UNCHANGED)
            img = np.asarray(img).astype(np.int8)
            (n, _) = np.histogram(img.flat, bins=bins, normed=False)

            s_pixel += n
            result = {"name": image_path, "count": s_pixel}
            result_queue.put(result)

    def data_val(self, data_path, result_queue, cou, heigh, i_dict):
        dicts = {}
        while 1:
            if result_queue.empty():
                break
            result = result_queue.get()
            cls = int(len(self_full_labels))
            for i in range(cls):
                if i not in dicts:
                    dicts[i] = [0, 0, []]
                dicts[i][0] += result["count"][i]
                if result["count"][i] != 0:
                    dicts[i][1] += 1
                    dicts[i][2].append(result["name"])


        name = os.path.basename(data_path)
        type_name = name.split('_')[-1][:-4]
        if i_dict:
            for k, v in i_dict.items():
                if k not in dicts:
                    v[type_name] = ["0%", "0", str(cou), []]
                else:
                    v[type_name] = [
                        str(round(dicts[k][0] * 100 * 1.0 / (heigh * 2448 * cou), 6)) + "%",
                        str(dicts[k][1]),
                        str(cou),
                        dicts[k][2]
                    ]
        else:
            for label in self_full_labels:
                i_dict[label.id] = {}
                i_dict[label.id]["name"] = label.name
                if label.id not in dicts:
                    i_dict[label.id][type_name] = ["0%", "0", str(cou), []]
                else:
                    i_dict[label.id][type_name] = [
                        str(round(dicts[label.id][0] * 100 * 1.0 / (heigh * 2448 * cou), 6)) + "%",
                        str(dicts[label.id][1]),
                        str(cou),
                        dicts[label.id][2]
                    ]


def do_data_count(data_dir, image_type):
    file_list = os.listdir(data_dir)
    if image_type == "full":
        heigh = 2048
    else:
        heigh = 1024
    dicts = {}
    for i in file_list:
        if i.endswith('.lst'):
            t1 = time.time()
            data_path = os.path.join(data_dir, i)
            track_handler = TrackSamImage()
            manager = multiprocessing.Manager()
            download_queue = manager.Queue()
            result_queue = manager.Queue()
            with open(data_path, 'r') as f:
                while 1:
                    file_path = f.readline()
                    if file_path:

                        image_path = file_path.split('\t')[3].strip()
                        download_task = DownloadSamTask(image_path=image_path)
                        download_queue.put(download_task)
                    else:
                        break
            cou = download_queue.qsize()
            for x in range(32):
                download_task = DownloadSamTask(image_path=None, exit_flag=True)
                download_queue.put(download_task)
            download_procs = []

            for x in range(32):
                download_proc = multiprocessing.Process(
                    target=track_handler.download_image, args=(download_queue, result_queue))
                download_proc.daemon = True
                download_procs.append(download_proc)

            for proc in download_procs:
                proc.start()

            for proc in download_procs:
                proc.join()
            t2 = time.time() - t1
            print t2
            track_handler.data_val(data_path, result_queue, cou, heigh, dicts)
    with open(os.path.join(data_dir, 'class.json'), 'w') as f:
        f.write(json.dumps(dicts))
    print "success"


if __name__ == '__main__':
    # data_path = sys.argv[1]
    # image_type = sys.argv[2]
    # do_data_count(data_path, image_type)
    do_data_count("/data/deeplearning/train_platform/data/beijing_sihuan", "half")
