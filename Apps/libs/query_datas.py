# -*-coding:utf-8-*-
import json


def filter_data(data_li, imgrange="0", city=None, label_info=None, tag_info=None, pacid=None):
    re_data = []
    if city and label_info and tag_info and pacid:
        nn_data = filter(
            lambda x: x["imgrange"] == imgrange and x["city"] == city and x["label_info"] == label_info and x["pacid"] == pacid,
            data_li)
        for data in nn_data:
            for tag in tag_info:
                if str(tag) in json.loads(data["tag_info"]):
                    re_data.append(data)
                    break

    elif not city and not label_info and not tag_info and pacid:
        nn_data = filter(lambda x: x["imgrange"] == imgrange and x["pacid"] == pacid, data_li)
        re_data = nn_data

    elif city and label_info and not tag_info and pacid:
        nn_data = filter(
            lambda x: x["imgrange"] == imgrange and x["city"] == city and x["label_info"] == label_info and x["pacid"] == pacid,
            data_li)
        re_data = nn_data

    elif city and not label_info and not tag_info and pacid:
        nn_data = filter(lambda x: x["imgrange"] == imgrange and x["city"] == city and x["pacid"] == pacid, data_li)
        re_data = nn_data

    elif city and not label_info and tag_info and pacid:
        nn_data = filter(lambda x: x["imgrange"] == imgrange and x["city"] == city and x["pacid"] == pacid, data_li)
        for data in nn_data:
            for tag in tag_info:
                if str(tag) in json.loads(data["tag_info"]):
                    re_data.append(data)
                    break

    elif not city and label_info and tag_info and pacid:
        nn_data = filter(lambda x: x["imgrange"] == imgrange and x["label_info"] == label_info and x["pacid"] == pacid,
                         data_li)
        for data in nn_data:
            for tag in tag_info:
                if str(tag) in json.loads(data["tag_info"]):
                    re_data.append(data)
                    break

    elif not city and not label_info and tag_info and pacid:
        nn_data = filter(lambda x: x["imgrange"] == imgrange and x["pacid"] == pacid, data_li)
        for data in nn_data:
            for tag in tag_info:
                if str(tag) in json.loads(data["tag_info"]):
                    re_data.append(data)
                    break

    elif not city and label_info and not tag_info and pacid:
        nn_data = filter(lambda x: x["imgrange"] == imgrange and x["label_info"] == label_info and x["pacid"] == pacid,
                         data_li)
        re_data = nn_data

    elif city and label_info and tag_info and not pacid:
        nn_data = filter(lambda x: x["imgrange"] == imgrange and x["city"] == city and x["label_info"] == label_info,
                         data_li)
        for data in nn_data:
            for tag in tag_info:
                if str(tag) in json.loads(data["tag_info"]):
                    re_data.append(data)
                    break

    elif not city and not label_info and not tag_info and not pacid:
        nn_data = filter(lambda x: x["imgrange"] == imgrange, data_li)
        re_data = nn_data

    elif city and label_info and not tag_info and not pacid:
        nn_data = filter(lambda x: x["imgrange"] == imgrange and x["city"] == city and x["label_info"] == label_info,
                         data_li)
        re_data = nn_data

    elif city and not label_info and not tag_info and not pacid:
        nn_data = filter(lambda x: x["imgrange"] == imgrange and x["city"] == city, data_li)
        re_data = nn_data

    elif city and not label_info and tag_info and not pacid:
        nn_data = filter(lambda x: x["imgrange"] == imgrange and x["city"] == city, data_li)
        for data in nn_data:
            for tag in tag_info:
                if str(tag) in json.loads(data["tag_info"]):
                    re_data.append(data)
                    break

    elif not city and label_info and tag_info and not pacid:
        nn_data = filter(lambda x: x["imgrange"] == imgrange and x["label_info"] == label_info, data_li)
        for data in nn_data:
            for tag in tag_info:
                if str(tag) in json.loads(data["tag_info"]):
                    re_data.append(data)
                    break

    elif not city and not label_info and tag_info and not pacid:
        nn_data = filter(lambda x: x["imgrange"] == imgrange, data_li)
        for data in nn_data:
            for tag in tag_info:
                if str(tag) in json.loads(data["tag_info"]):
                    re_data.append(data)
                    break

    elif not city and label_info and not tag_info and not pacid:
        nn_data = filter(lambda x: x["imgrange"] == imgrange and x["label_info"] == label_info, data_li)
        re_data = nn_data

    return re_data
