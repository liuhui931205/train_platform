# -*- coding:utf-8 -*-
import os
import mxnet as mx

from collections import namedtuple

Label = namedtuple(
    'Label', ['en_name', 'id', 'categoryId', 'color', 'name'])

Labels = namedtuple(
    'Labels', ['categoryId', 'color', 'name'])

Label1 = namedtuple(
    'Label', ['id', 'categoryId', 'label', 'name'])

Label2 = namedtuple(
    'Label', ['className', 'en_name', 'categoryId', 'trainId', 'color'])
# self_road_chn_labels
self_road_chn_labels = {
    Label('other', 0, 0, (64, 64, 32), u'其他'),
    Label('ignore', 1, 1, (0, 0, 0), u'Ignore'),
    Label('lane_w', 2, 2, (255, 0, 0), u'车道标线-白色'),
    Label('left', 3, 3, (255, 192, 203), u'左侧道路边缘线'),
    Label('right', 4, 4, (139, 0, 139), u'右侧道路边缘线'),
    Label('v_slow', 5, 5, (32, 128, 192), u'纵向减速标线'),
    Label('bus_lane', 6, 6, (192, 128, 255), u'专用车道标线'),
    Label('stop', 7, 7, (255, 128, 64), u'停止线'),
    Label('slow_let', 8, 8, (0, 255, 255), u'减速让行标线'),
    Label('slow_zone', 9, 9, (128, 128, 255), u'减速标线/减速带'),
    Label('sidewalk', 10, 10, (128, 192, 192), u'人行横道'),
    Label('connection', 11, 11, (128, 128, 192), u'路面连接带'),
    Label('stop_station', 12, 12, (240, 128, 128), u'停靠站标线'),
    Label('in_out', 13, 13, (128, 128, 0), u'出入口标线'),
    Label('symbol', 14, 14, (0, 0, 255), u'文字符号类'),
    Label('fish_lane', 15, 15, (0, 255, 0), u'导流线（鱼刺线）'),
    Label('stop_gird', 16, 16, (255, 255, 0), u'停止网格标线'),
    Label('distance', 17, 17, (255, 128, 255), u'车距确认线'),
    Label('road', 18, 18, (192, 192, 192), u'道路'),
    Label('objects', 19, 19, (128, 0, 0), u'车辆及路面上其他物体'),
    Label('curb', 20, 20, (0, 139, 139), u'虚拟车道线-路缘石'),
    Label('fence', 21, 21, (255, 106, 106), u'虚拟车道线-防护栏'),
    Label('virtual', 22, 22, (118, 180, 254), u'虚拟车道线-其他'),
    Label('tide_lane', 23, 23, (75, 0, 130), u'潮汐车道线'),
    Label('left_wait', 24, 24, (144, 238, 144), u'左弯待转区线'),
    Label('guide_lane', 25, 25, (0, 255, 127), u'可变导向车道线'),
    Label('lane_y', 26, 26, (255, 165, 0), u'车道标线-黄色'),
    Label('hump', 27, 27, (72, 61, 139), u'减速丘'),
}

self_full_labels = {
    Label('other', 0, 0, (64, 64, 32), u'其他'),
    Label('ignore', 1, 1, (0, 0, 0), u'Ignore'),
    Label('lane', 2, 2, (255, 0, 0), u'车道标线-白色'),
    Label('left', 3, 3, (255, 192, 203), u'左侧道路边缘线'),
    Label('right', 4, 4, (139, 0, 139), u'右侧道路边缘线'),
    Label('v_slow', 5, 5, (32, 128, 192), u'纵向减速标线'),
    Label('bus_lane', 6, 6, (192, 128, 255), u'专用车道标线'),
    Label('stop', 7, 7, (255, 128, 64), u'停止线'),
    Label('slow_let', 8, 8, (0, 255, 255), u'减速让行标线'),
    Label('slow_zone', 9, 9, (128, 128, 255), u'减速标线/减速带'),
    Label('sidewalk', 10, 10, (128, 192, 192), u'人行横道'),
    Label('connection', 11, 11, (128, 128, 192), u'路面连接带'),
    Label('stop_station', 12, 12, (240, 128, 128), u'停靠站标线'),
    Label('in_out', 13, 13, (128, 128, 0), u'出入口标线'),
    Label('symbol', 14, 14, (0, 0, 255), u'文字符号类'),
    Label('fish_lane', 15, 15, (0, 255, 0), u'导流线（鱼刺线）'),
    Label('stop_gird', 16, 16, (255, 255, 0), u'停止网格标线'),
    Label('distance', 17, 17, (255, 128, 255), u'车距确认线'),
    Label('road', 18, 18, (192, 192, 192), u'道路'),
    Label('objects', 19, 19, (128, 0, 0), u'车辆及路面上其他物体'),
    Label('curb', 20, 20, (0, 139, 139), u'虚拟车道线-路缘石'),
    Label('fence', 21, 21, (255, 106, 106), u'虚拟车道线-防护栏'),
    Label('virtual', 22, 22, (118, 180, 254), u'虚拟车道线-其他'),
    Label('tide_lane', 23, 23, (75, 0, 130), u'潮汐车道线'),
    Label('left_wait', 24, 24, (144, 238, 144), u'左弯待转区线'),
    Label('guide_lane', 25, 25, (0, 255, 127), u'可变导向车道线'),
    Label('lane_y', 26, 26, (255, 165, 0), u'车道标线-黄色'),
    Label('hump', 27, 27, (72, 61, 139), u'减速丘'),
    Label('warning', 28, 28, (255, 0, 255), u'警告标示牌'),
    Label('prohibition', 29, 29, (220, 20, 60), u'禁令标志牌'),
    Label('instructive', 30, 30, (255, 215, 0), u'指示标志牌'),
    Label('highway', 31, 31, (0, 0, 128), u'高速公路指路标志牌'),
    Label('directing', 32, 32, (153, 102, 51), u'普通道路指路标志牌'),
    Label('tourist', 33, 33, (102, 102, 0), u'旅游区标志牌'),
    Label('assist', 34, 34, (0, 255, 102), u'辅助标志牌'),
    Label('task', 35, 35, (0, 102, 0), u'作业区标志牌'),
    Label('other', 36, 36, (204, 255, 0), u'其他交通标志牌'),
    Label('pole', 37, 37, (255, 140, 0), u'杆'),
    Label('sky', 38, 38, (135, 206, 235), u'天空'),
    Label('curbs', 39, 39, (127, 255, 0), u'路缘石'),
    Label('barrier', 40, 40, (0, 250, 154), u'凸型屏障'),
    Label('fences', 41, 41, (5, 107, 47), u'防护栏'),
    Label('light', 42, 42, (255, 69, 0), u'灯'),
    Label('cover', 43, 43, (178, 34, 34), u'井盖'),
    Label('fence', 44, 44, (65, 105, 225), u'可移动栅栏'),
    Label('reflector', 45, 45, (50, 205, 50), u'反光标'),
    Label('old_line', 46, 46, (255, 20, 147), u'旧车道线'),
    Label('w_dot_line', 47, 47, (127, 255, 212), u'车道标线-白虚线'),
    Label('y_dot_line', 48, 48, (199, 21, 133), u'车道标线-黄虚线'),
    Label('point', 49, 49, (139, 69, 19), u'鱼刺线端点'),
    Label('other_rail', 50, 50, (0, 191, 255), u'其他栏杆'),
    Label('acoustic_barrier', 51, 51, (238, 232, 170), u'隔音护栏'),
    Label('wall', 52, 52, (0, 0, 139), u'墙护栏'),
    Label('fixed_fence', 53, 53, (205, 92, 92), u'固定围栏'),
    Label('flush_curb', 54, 54, (70, 130, 180), u'平缘石'),
    Label('curb_plane', 55, 55, (105, 105, 105), u'路缘石-平面'),
    Label('overhead', 56, 56, (255, 250, 240), u'顶部区域-垂面'),
    Label('overhead_plane', 57, 57, (0, 206, 209), u'顶部区域-平面'),
    Label('pillar', 58, 58, (244, 164, 96), u'立柱'),
    Label('wall_tunnel', 59, 59, (47, 79, 79), u'隧道墙'),
    Label('traffic_light', 60, 60, (143, 188, 143), u'交通信号灯'),
    Label('traffic_light_top', 61, 61, (255, 99, 71), u'信号灯灯头'),
    Label('passenger_car', 62, 62, (210, 105, 30), u'客车(公交车或大巴)'),
    Label('truck', 63, 63, (245, 222, 179), u'卡车'),
    Label('movable_obj', 64, 64, (218, 165, 32), u'路面上可移动物体'),
    Label('privileged_vehicles', 65, 65, (85, 107, 47), u'特权车'),
    Label('electrocar', 66, 66, (152, 251, 152), u'摩托车或电动车'),
    Label('bicycle', 67, 67, (72, 209, 204), u'自行车'),
    Label('pedestrian', 68, 68, (123, 104, 238), u'行人(小孩)'),
    Label('sidewalk', 69, 69, (188, 143, 143), u'人行道'),
    Label('trash_can', 70, 70, (176, 196, 222), u'垃圾桶'),
    Label('advertising_board', 71, 71, (230, 230, 250), u'广告牌'),
    Label('construction', 72, 72, (119, 136, 153), u'建筑物'),
    Label('vegetation', 73, 73, (189, 183, 107), u'草木'),
    Label('tricycle', 74, 74, (221, 160, 221), u'三轮车'),
    Label('other_lines', 75, 75, (112, 128, 144), u'路面其他车道线'),
    Label('bull_barrels', 76, 76, (250, 128, 114), u'防撞桶'),
    Label('car_head', 77, 77, (0, 128, 128), u'采集车车头'),
    Label('r_l_line', 97, 97, (205, 0, 0), u'右黄线'),
    Label('l_y_line', 98, 98, (205, 90, 106), u'左黄线'),
    Label('dot_line', 99, 99, (238, 130, 238), u'虚实线_虚线'),
}

seg_label = {
    Labels(255, (0, 0, 0), u'Ignore'),
    Labels(254, (0, 139, 139), u'虚拟车道线-路缘石'),
    Labels(0, (255, 69, 0), u'车道标线-白色'),
    Labels(1, (255, 192, 203), u'左侧道路边缘线'),
    Labels(2, (139, 0, 139), u'右侧道路边缘线'),
    Labels(3, (32, 128, 192), u'纵向减速标线'),
    Labels(4, (192, 128, 255), u'专用车道标线'),
    Labels(5, (255, 128, 64), u'停止线'),
    Labels(6, (128, 128, 0), u'出入口标线'),
    Labels(7, (0, 0, 255), u'文字符号类'),
    Labels(8, (0, 255, 0), u'导流线（鱼刺线）'),
    Labels(9, (128, 64, 128), u'道路'),
    Labels(10, (128, 0, 0), u'车辆及路面上其他物体'),
    Labels(11, (128, 128, 192), u'路面连接带'),
    Labels(12, (64, 64, 32), u'其他'),
    Labels(14, (127, 255, 0), u'路缘石'),
    Labels(13, (255, 20, 147), u'旧车道线'),
    Labels(15, (50, 205, 50), u'反光标'),
    Labels(16, (135, 206, 235), u'天空'),
    Labels(17, (255, 0, 255), u'路牌'),
    Labels(18, (255, 140, 0), u'杆'),
    Labels(19, (0, 191, 255), u'栏杆'),
    Labels(20, (255, 250, 240), u'顶部区域-垂面'),
    Labels(21, (0, 206, 209), u'顶部区域-平面'),
    Labels(22, (105, 105, 105), u'路缘石-平面'),
    Labels(23, (255, 99, 71), u'信号灯灯头'),
    Labels(24, (210, 105, 30), u'客车(公交车或大巴)'),
    Labels(25, (245, 222, 179), u'卡车'),
    Labels(26, (218, 165, 32), u'路面上可移动物体'),
    Labels(27, (85, 107, 47), u'特权车'),
    Labels(28, (152, 251, 152), u'摩托车或电动车'),
    Labels(29, (72, 209, 204), u'自行车'),
    Labels(30, (123, 104, 238), u'行人(小孩)'),
    Labels(31, (188, 143, 143), u'人行道'),
    Labels(32, (176, 196, 222), u'垃圾桶'),
    Labels(33, (230, 230, 250), u'广告牌'),
    Labels(34, (119, 136, 153), u'建筑物'),
    Labels(35, (189, 183, 107), u'草木'),
    Labels(36, (221, 160, 221), u'三轮车'),
    Labels(37, (0, 0, 139), u'墙护栏'),
    Labels(38, (178, 34, 34), u'井盖'),
    Labels(39, (65, 105, 225), u'可移动栅栏'),
    Labels(40, (255, 165, 0), u'车道标线-黄色'),
    Labels(41, (128, 192, 192), u'人行横道'),
    Labels(42, (128, 128, 255), u'减速标线/减速带'),
    Labels(43, (255, 255, 0), u'停止网格标线'),
    Labels(44, (127, 255, 212), u'车道标线-白虚线'),
    Labels(45, (238, 130, 238), u'虚实线_虚线'),
    Labels(46, (255, 128, 255), u'车距确认线'),
    Labels(47, (112, 128, 144), u'路面其他车道线'),
    Labels(48, (250, 128, 114), u'防撞桶'),
    Labels(49, (205, 90, 106), u'左黄线'),
    Labels(50, (205, 0, 0), u'右黄线'),
    Labels(51, (0, 250, 154), u'凸型屏障'),
    Labels(52, (5, 107, 47), u'防护栏'),
    Labels(53, (238, 232, 170), u'隔音护栏'),
    Labels(54, (205, 92, 92), u'固定围栏'),
    Labels(55, (0, 128, 128), u'采集车车头'),
}

arrow_labels = {
    Label1(0,    0,      "0",    u"直行"),
    Label1(1,    1,      "1",    u"左转"),
    Label1(2,    2,      "2",    u"右转"),
    Label1(3,    3,      "3",    u"掉头"),
    Label1(4,    4,      "4",    u"左汇入"),
    Label1(5,    5,      "5",    u"直行加左转"),
    Label1(6,    6,      "6",    u"直行加右转"),
    Label1(7,    7,      "7",    u"直行加掉头"),
    Label1(8,    8,      "8",    u"左转加掉头"),
    Label1(9,    9,      "9",    u"左右转"),
    Label1(10,   10,     "10",   u"右转加掉头"),
    Label1(11,   11,     "11",   u"右汇入"),
    Label1(12,   12,     "12",   u"符号或文字或其他标志"),
}

scene_labels = {
    Label1(0,    0,      "0",    u"高速"),
    Label1(1,    1,      "1",    u"城市道路"),
    Label1(2,    2,      "2",    u"隧道"),
    Label1(3,    3,      "3",    u"夜间"),
}

object_labels_sign = {
    Label2('background',     "background",     "__background__",  0, (0, 0, 0)),
    Label2('traffic_signs',  "traffic_signs",  "1",    1, (0, 0, 255)),
    Label2('traffic_lights', "traffic_lights", "2",    2, (255, 0, 255)),
}

traffic_signs = [24, 25, 26, 27, 28, 29, 30, 31, 32]
traffic_lights = [60]
vehicles = [12, 62, 63, 65, 66, 67, 74]
pedestrian = [68]


def get_file(file_dir, file_list, src_len):
    files = os.listdir(file_dir)
    for _file in files:
        _dir = os.path.join(file_dir, _file)
        if os.path.isdir(_dir):
            get_file(_dir, file_list, src_len)
        else:
            _name_list = str(_file).split(".")
            if len(_name_list) < 2:
                continue
            # _file_name = _name_list[len(_name_list) - 2]
            _file_ext = _name_list[len(_name_list) - 1]

            if _file_ext not in ["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"]:
                continue
            _file_path = _dir[src_len + 1:]
            if isinstance(file_list, list):
                file_list.append(_file_path)


class Task(object):
    def __init__(self, package_index, src_path, dest_path, dest_label, exit_flag=False):
        self.package_index = package_index
        self.src_path = src_path
        self.dest_path = dest_path
        self.dest_label = dest_label
        self.exit_flag = exit_flag


class RemoteTask(object):
    def __init__(self, package_index, src_path, dest_path, exit_flag=False):
        self.package_index = package_index
        self.src_path = src_path
        self.dest_path = dest_path
        self.exit_flag = exit_flag


class OnlineTask(object):
    def __init__(self, track_point_id, task_id, image_data, label_data, exit_flag=False):
        self.track_point_id = track_point_id
        self.task_id = task_id
        self.image_data = image_data
        self.label_data = label_data
        self.exit_flag = exit_flag

