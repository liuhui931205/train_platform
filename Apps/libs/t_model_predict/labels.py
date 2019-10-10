#!/usr/bin/env python
#encoding=utf-8

from collections import namedtuple

Label = namedtuple(
    'Label', ['className', 'alias', 'categoryId', 'trainId', 'color'])

object_labels = {
    Label('background',     "background",     "__background__",  0, (0, 0, 0)),
    Label('vehicles',       "vehicles",       "0",    1, (255, 0, 0)),
    Label('pedestrian',     "pedestrian",     "1",    2, (0, 255, 0)),
    Label('traffic_signs',  "traffic_signs",  "2",    3, (0, 0, 255)),
    Label('traffic_lights', "traffic_lights", "3",    4, (255, 0, 255)),
}

object_labels_sign = {
    Label('background',     "background",     "__background__",  0, (0, 0, 0)),
    Label('traffic_signs',  "traffic_signs",  "1",    1, (0, 0, 255)),
    Label('traffic_lights', "traffic_lights", "2",    2, (255, 0, 255)),
}