# coding: utf-8
from sqlalchemy import Column, DateTime, Integer, String, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()
metadata = Base.metadata


class TrainTask(Base):
    __tablename__ = 'train_task'

    id = Column(Integer, primary_key=True)
    task_id = Column(String(255))
    task_name = Column(String(255))
    network_path = Column(String(255))
    task_describe = Column(String(255))
    image_type = Column(String(255))
    data_path = Column(String(255))
    category_num = Column(String(255))
    iter_num = Column(String(255))
    learning_rate = Column(String(255))
    batch_size = Column(String(255))
    steps = Column(String(255))
    gpus = Column(String(255))
    model = Column(String(255))
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    net_desc_id = Column(Integer, ForeignKey('network_describe.id'))
    data_desc_id = Column(Integer, ForeignKey('datas.id'))
    net_desc = relationship('NetworkDescribe')
    data_desc = relationship('Datas')
    machine_id = Column(String(255))
    type = Column(String(255))
    weights = Column(String(255))
    parallel_bool = Column(String(255))
    status = Column(String(255))

    def to_dict(self):
        desc_dict = {
            'id': self.id,
            'task_id': self.task_id,
            'task_name': self.task_name,
            'network_path': self.network_path,
            'category_num': self.category_num,
            'data_path': self.data_path,
            'iter_num': self.iter_num,
            'learning_rate': self.learning_rate,
            'steps': self.steps,
            'batch_size': self.batch_size,
            'gpus': self.gpus,
            'model': self.model,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'net_desc': self.net_desc.net_describe,
            'task_desc': self.task_describe,
            'data_desc': self.data_desc.data_describe,
            'status': self.status,
            'data_name': self.data_desc.data_name,
            'machine_id': self.machine_id,
            'type': self.type,
            'weights': self.weights,
            'image_type': self.image_type,
            "parallel_bool": self.parallel_bool
        }
        return desc_dict


class NetworkDescribe(Base):
    __tablename__ = 'network_describe'

    id = Column(Integer, primary_key=True)
    task_id = Column(String(255))
    net_name = Column(String(255))
    src_network = Column(String(255))
    net_describe = Column(String(255))
    type = Column(String(255))
    status = Column(String(255))

    def to_net_dict(self):
        net_desc_dict = {
            'id': self.id,
            'task_id': self.task_id,
            'status': self.status,
            'net_name': self.net_name,
            'net_describe': self.net_describe,
            'type': self.type
        }
        return net_desc_dict


class Datas(Base):
    __tablename__ = 'datas'

    id = Column(Integer, primary_key=True)
    task_id = Column(String(255))
    data_name = Column(String(255))
    data_type = Column(String(255))
    images_type = Column(String(255))
    type = Column(String(255))
    train = Column(String(255))
    val = Column(String(255))
    test = Column(String(255))
    sour_data = Column(String(255))
    data_describe = Column(String(255))
    status = Column(String(255))
    machine_id = Column(String(255))

    def to_data_dict(self):
        data_desc_dict = {
            'id': self.id,
            'task_id': self.task_id,
            'data_type': self.data_type,
            'train': self.train,
            'val': self.val,
            'test': self.test,
            'machine_id': self.machine_id,
            'sour_data': self.sour_data,
            'status': self.status,
            'data_name': self.data_name,
            'data_describe': self.data_describe,
            'type': self.type
        }
        return data_desc_dict


class Released_Models(Base):
    __tablename__ = 'released_models'

    id = Column(Integer, primary_key=True)
    task_id = Column(String(255))
    tasks_name = Column(String(255))
    version = Column(String(255))
    model_name = Column(String(255))
    env = Column(String(255))
    adcode = Column(String(255))
    desc = Column(String(255))
    time = Column(String(255))
    type = Column(String(255))
    status = Column(String(255))

    def to_models_dict(self):
        model_dict = {
            'id': self.id,
            'task_id': self.task_id,
            'tasks_name': self.tasks_name,
            'version': self.version,
            'model_name': self.model_name,
            'env': self.env,
            'adcode': self.adcode,
            'desc': self.desc,
            'time': self.time,
            'type': self.type,
            'status': self.status
        }
        return model_dict


class Evaluate_Models(Base):
    __tablename__ = 'evaluate_models'

    id = Column(Integer, primary_key=True)
    task_id = Column(String(255))
    sour_dir = Column(String(500))
    gpus = Column(String(255))
    dest_dir = Column(String(255))
    single_gpu = Column(String(255))
    model = Column(String(500))
    status = Column(String(255))
    host = Column(String(255))

    def to_model_dict(self):
        model_dict = {
            'id': self.id,
            'task_id': self.task_id,
            'sour_dir': self.sour_dir,
            'gpus': self.gpus,
            'dest_dir': self.dest_dir,
            'single_gpu': self.single_gpu,
            'model': self.model,
            'status': self.status,
            "host": self.host
        }
        return model_dict


class Auto_select(Base):
    __tablename__ = 'auto_select'
    id = Column(Integer, primary_key=True)
    task_type = Column(String(255))
    task_id = Column(String(255))
    output_dir = Column(String(255))
    gpus = Column(String(255))
    sele_ratio = Column(String(255))
    weights_dir = Column(String(255))
    track_file = Column(String(255))
    task_file = Column(String(255))
    isshuffle = Column(String(255))
    status = Column(String(255))

    def to_dict(self):
        select_dict = {
            'task_id': self.task_id,
            'task_type': self.task_type,
            'output_dir': self.output_dir,
            'gpus': self.gpus,
            'sele_ratio': self.sele_ratio,
            'weights_dir': self.weights_dir,
            'track_file': self.track_file,
            'task_file': self.task_file,
            'isshuffle': self.isshuffle,
            'status': self.status
        }
        return select_dict


class OfflineImport(Base):
    __tablename__ = 'offline_datas'

    id = Column(Integer, primary_key=True)
    task_id = Column(String(255))
    roadelement = Column(String(255))
    source = Column(String(255))
    author = Column(String(255))
    annotype = Column(String(255))
    datakind = Column(String(255))
    city = Column(String(255))
    imgoprange = Column(String(255))
    status = Column(String(255))

    def to_dict(self):
        data_dict = {
            'id': self.id,
            'task_id': self.task_id,
            'roadelement': self.roadelement,
            'source': self.source,
            'author': self.author,
            'annotype': self.annotype,
            'datakind': self.datakind,
            'city': self.city,
            'imgoprange': self.imgoprange,
            'status': self.status
        }
        return data_dict


class LineDownload(Base):
    __tablename__ = 'linedown_datas'

    id = Column(Integer, primary_key=True)
    task_id = Column(String(255))
    taskid_start = Column(String(255))
    taskid_end = Column(String(255))
    dest = Column(String(255))
    status = Column(String(255))

    def to_dict(self):
        data_dict = {
            'task_id': self.task_id,
            'taskid_start': self.taskid_start,
            'taskid_end': self.taskid_end,
            'dest': self.dest,
            'status': self.status
        }
        return data_dict


class TaskDivide(Base):
    __tablename__ = 'task_divide'

    id = Column(Integer, primary_key=True)
    task_id = Column(String(255))
    version = Column(String(255))
    step = Column(String(255))
    types = Column(String(255))
    status = Column(String(255))

    def to_dict(self):
        data_dict = {
            'task_id': self.task_id,
            'version': self.version,
            'step': self.step,
            'types': self.types,
            'status': self.status
        }
        return data_dict


class LabelProcess(Base):
    __tablename__ = 'label_process'

    id = Column(Integer, primary_key=True)
    task_id = Column(String(255))
    version = Column(String(255))
    name = Column(String(255))
    types = Column(String(255))
    color_info = Column(String(255))
    status = Column(String(255))

    def to_dict(self):
        data_dict = {
            'task_id': self.task_id,
            'version': self.version,
            'name': self.name,
            'types': self.types,
            'color_info': self.color_info,
            'status': self.status,
        }
        return data_dict


class CheckDatas(Base):
    __tablename__ = 'check_datas'

    id = Column(Integer, primary_key=True)
    task_id = Column(String(255))
    task_name = Column(String(255))
    weights_dir = Column(String(255))
    status = Column(String(255))

    def to_dict(self):
        data_dict = {
            'task_id': self.task_id,
            'task_name': self.task_name,
            'weights_dir': self.weights_dir,
            'status': self.status
        }
        return data_dict


class Confidence_Datas(Base):
    __tablename__ = 'confidence_datas'

    id = Column(Integer, primary_key=True)
    origin_whole_con = Column(String(255))
    whole_con = Column(String(255))
    origin_cls_con = Column(String(500))
    model = Column(String(500))
    trackpointid = Column(String(500))
    task_name = Column(String(500))

    def to_dict(self):
        data_dict = {
            'origin_whole_con': self.origin_whole_con,
            'whole_con': self.whole_con,
            'origin_cls_con': self.origin_cls_con,
            'trackpointid': self.trackpointid,
            'model': self.model,
            'task_name': self.task_name
        }
        return data_dict


class LabelData(Base):
    __tablename__ = 'data_manage'
    id = Column(Integer, primary_key=True)
    time_info = Column(String(255))
    trackpointid = Column(String(255))
    pacid = Column(String(255))
    imgrange = Column(String(255))
    city = Column(String(255))
    label_info = Column(String(255))
    tag_info = Column(String(4000))

    def to_dict(self):
        data_dict = {
            'time_info': self.time_info,
            'trackpointid': self.trackpointid,
            'imgrange': self.imgrange,
            'city': self.city,
            'pacid': self.pacid,
            'label_info': self.label_info,
            'tag_info': self.tag_info
        }
        return data_dict


class Schedulefrom(Base):
    __tablename__ = 'progress'
    id = Column(Integer, primary_key=True)
    id_code = Column(String(255))
    task_id = Column(String(255))
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    picid = Column(String(255))
    statue = Column(String(255))

    def to_dict(self):
        data_dict = {
            "id_code": self.id_code,
            'task_id': self.task_id,
            'picid': self.picid,
            'start_time': self.start_time,
            "end_time": self.end_time,
            'statue': self.statue
        }
        return data_dict


class ScoreTasks(Base):
    __tablename__ = 'score_tasks'

    id = Column(Integer, primary_key=True)
    task_id = Column(String(255))
    area_name = Column(String(255))
    scence_name = Column(String(255))
    gpus = Column(String(255))
    png_path = Column(String(255))
    task_name = Column(String(255))
    img = Column(String(255))
    score = Column(String(255))
    standard = Column(String(255))
    status = Column(String(255))
    host = Column(String(255))

    def to_data_dict(self):
        data_desc_dict = {
            'id': self.id,
            'task_id': self.task_id,
            'area_name': self.area_name,
            'scence_name': self.scence_name,
            'gpus': self.gpus,
            'task_name': self.task_name,
            'img': self.img,
            'status': self.status,
            'standard': self.standard,
            'png_path': self.png_path,
            'score': self.score,
            'host': self.host
        }
        return data_desc_dict
