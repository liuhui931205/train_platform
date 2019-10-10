
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
    def __init__(self, label_data):
        super(KdSegId2Color, self).__init__()
        self._cmap = np.zeros((256, 3), dtype=np.uint8)
        for label in label_data:
            # if label.id in (-1, 255):
            #     continue
            self._cmap[int(label["categoryId"])] = tuple(label["color"])

    def __call__(self, data):
        """Functor"""
        return self._cmap[data]