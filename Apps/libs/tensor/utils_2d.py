"""Putting 2D image utils here."""

import numpy as np


# Kd data, seg-id and seg-id color
class KdSegId(object):
    def __init__(self, label_map_data):
        """Initializer."""
        super(KdSegId, self).__init__()
        self._label_2_id = np.full((256,), 255)

        for label in label_map_data:
            if int(label[0]) in (-1, 255):
                continue
            self._label_2_id[int(label[0])] = int(label[1])

    def __call__(self, data):
        """Functor"""
        return self._label_2_id[data]


class KdSegId2Color(object):

    def __init__(self, label_map):
        super(KdSegId2Color, self).__init__()
        self._cmap = np.zeros((256, 3), dtype=np.uint8) - 1
        # kd_road_labels = sorted(kd_helper.kd_road_labels, key=lambda d: d[1], reverse=True)
        # for k, v in label_map.items():
        #     if int(v[1]) in (-1, 255):
        #         continue
        #     self._cmap[int(k)] = tuple(v[2])
        for label in label_map:
            # if label.id in (-1, 255):
            #     continue
            self._cmap[label["categoryId"]] = tuple(label["color"])

    def __call__(self, data):
        """Functor"""
        return self._cmap[data]


