# -*-coding:utf-8-*-

import numpy as np
import cv2
import os
from PIL import Image
import mxnet as mx
import time
import copy
import tensorrt as trt
from Apps.libs.s_model_predict.util import load_weights
from Apps.libs.s_model_predict import symbol

import pycuda.driver as cuda

# from .seg_labels import kd_germany_test_labels as kd_road_deploy_labels

# from cls13_large_20180507.kd_helper import kd_road_deploy_labels
# from cls13_multilabel_20180528.kd_helper import kd_road_deploy_labels
# from models.arrow_classfication.model import ModelClassArrow
#   from models.virtual_lane.model import ModelVirtualLane

#   import global_variables


class TrtInfer(object):

    def __init__(self,
                 plan_file,
                 input_size,
                 cls,
                 output_size,
                 gpu_id,
                 argmax=True,
                 effective_c=0,
                 output_data_type=np.float32,
                 input_data_type=np.float32):
        if not os.path.exists(plan_file):
            raise ("plan file does not exist: {}".format(plan_file))
        self.output = np.empty(input_size[0] * cls * output_size[1] * output_size[0], dtype=output_data_type)
        self.temp_input = np.empty(1, dtype=input_data_type)
        self.output_c = cls
        self.output_w = output_size[1]
        self.output_h = output_size[0]
        self.batch_size = input_size[0]
        self.argmax = argmax
        self.effective_c = effective_c

        os.environ["CUDA_DEVICE"] = str(gpu_id)
        import pycuda.autoinit
        self.G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)
        self.engine = trt.utils.load_engine(self.G_LOGGER, plan_file)
        assert (self.engine.get_nb_bindings() == 2)
        # self._input_type = self._engine.data_type.input_type()

        self.context = self.engine.create_execution_context()
        self.d_input = cuda.mem_alloc(
            input_size[0] * input_size[1] * input_size[3] * input_size[2] * self.temp_input.dtype.itemsize)
        self.d_output = cuda.mem_alloc(
            input_size[0] * cls * output_size[1] * output_size[0] * self.output.dtype.itemsize)
        self.bindings = [int(self.d_input), int(self.d_output)]
        # self.g_prof = Profiler(1)
        # self.context.set_profiler(self.g_prof)

        self.stream = cuda.Stream()

    def destroy(self):
        self.context.destroy()
        self.engine.destroy()

    # def get_layer_time(self):
    #     self.g_prof.print_layer_times()

    def infer(self, data):
        # self.g_prof.reset()
        tic1 = time.time()
        cuda.memcpy_htod_async(self.d_input, data, self.stream)
        tic2 = time.time()
        self.context.enqueue(self.batch_size, self.bindings, self.stream.handle, None)
        tic3 = time.time()
        cuda.memcpy_dtoh_async(self.output, self.d_output, self.stream)
        tic4 = time.time()
        self.stream.synchronize()
        tic5 = time.time()
        print "kernel time: {},{},{},{}".format((tic2 - tic1), (tic3 - tic2), (tic4 - tic3), (tic5 - tic4))
        print "max score: {}".format(np.max(self.output))
        if (self.argmax):
            if (self.effective_c > 0):
                tic6 = time.time()
                temp = self.output.reshape(self.batch_size, self.output_c, self.output_h, self.output_w)
                temp = temp[0:1, 0:self.effective_c, 0:512, 0:1224]
                # result = [np.argmax(self.output.reshape(self.batch_size, self.output_c, self.output_h, self.output_w), axis=1)]
                result = [np.argmax(temp, axis=1)]
                # print "shape: {}".format(temp.shape)
                # print "max label: {}".format(np.max(result))
                print("give up, argmax time:{}".format(time.time() - tic6))
                return result
            else:
                tic6 = time.time()
                result = [
                    np.argmax(
                        self.output.reshape(self.batch_size, self.output_c, self.output_h, self.output_w), axis=1)
                ]
                # result = [np.argmax(temp, axis=1)]
                # print "shape: {}".format(temp.shape)
                print "max label: {}".format(np.max(result))
                print("argmax time:{}".format(time.time() - tic6))
                return result
        else:
            return [self.output.reshape(self.batch_size, self.output_h, self.output_w)]


class ModelResNetRoad:

    def __init__(self,
                 gpu_id=0,
                 model_path=None,
                 cls=None,
                 seg_label_li=None,
                 input_size=None,
                 output_size=None,
                 argmax=True,
                 effective_c=0):
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        cur_path = os.path.realpath(__file__)
        cur_dir = os.path.dirname(cur_path)
        cur_list = os.listdir(cur_dir)
        self.input_size = input_size
        self.output_size = output_size
        self.argmax = argmax
        self.effective_c = effective_c
        self.cls = cls
        self.weights = model_path
        self.seg_label_li = seg_label_li
        self.ignore_color = (0, 0, 0)
        # clr_dict = {l.name: l.color for l in kd_road_deploy_labels}
        # for name, color in clr_dict.items():
        # for k, v in self.seg_label_li.items():
        #     if v[0] == u'Ignore':
        #         self.ignore_color = tuple(v[1])
        #         break
        for i in self.seg_label_li:
            if i['name'] == u'Ignore':
                self.ignore_color = tuple(i['color'])
                break

        # label to color
        self.use_label_to_color = True
        self.label_to_color = np.zeros((256, 3), dtype=np.uint8)
        # for k, v in self.seg_label_li.items():
        #     if not int(k) in range(0, 256):
        #         continue
        #     self.label_to_color[int(k)] = (int(v[1][0]), int(v[1][1]), int(v[1][2]))
        for label in self.seg_label_li:
            if not int(label["categoryId"]) in range(0, 256):
                continue
            self.label_to_color[int(label["categoryId"])] = (label["color"][2], label["color"][1], label["color"][0])

        network, net_args, net_auxs = load_weights(self.weights)
        context = [mx.gpu(gpu_id)]
        self.mod = mx.mod.Module(network, context=context)

        self.result_shape = [1024, 2448]
        self.input_shape = [512, 1224]
        self.batch_data_shape = (1, 3, 512, 1224)
        provide_data = [("data", self.batch_data_shape)]
        self.batch_label_shape = (1, 512, 1224)
        provide_label = [("softmax_label", self.batch_label_shape)]
        self.mod.bind(provide_data, provide_label, for_training=False, force_rebind=True)
        self.mod.init_params(arg_params=net_args, aux_params=net_auxs)
        self._flipping = False

        self.batch_data = [mx.nd.empty(info[1]) for info in provide_data]
        self.batch_label = [mx.nd.empty(info[1]) for info in provide_label]

        symbol.cfg['workspace'] = 1024
        symbol.cfg['bn_use_global_stats'] = True

        # GPU Upsampling
        self.use_gpu_upsampling = True
        upsampling_sym = self.get_upsampling_sym()
        self.upsample_mod = mx.mod.Module(upsampling_sym, context=context, data_names=['data'], label_names=[])
        self.upsample_mod.bind(
            data_shapes=[('data', (1, int(self.cls), 512, 1224))],
            label_shapes=None,
            for_training=False,
            force_rebind=True)
        initializer = mx.init.Bilinear()
        self.upsample_mod.init_params(initializer=initializer)

        self.trt_infer = TrtInfer(
            self.weights,
            self.input_size,
            self.cls,
            self.output_size,
            gpu_id,
            argmax=self.argmax,
            effective_c=self.effective_c)

    # self.virtual_lane_net = ModelVirtualLane(gpu_id=gpu_id)

    def get_upsampling_sym(self):

        input_data = mx.symbol.Variable(name='data')
        upsampling = mx.symbol.UpSampling(
            input_data,
            scale=2,
            num_filter=int(self.cls),
            sample_type='bilinear',
            name="upsampling_preds",
            workspace=512)
        upsampling_sym = mx.sym.argmax(data=upsampling, axis=1)
        return upsampling_sym

    def do(self, image_data, dest_file=None):
        pred_data = None
        try:
            _time1 = time.time()
            image = np.asarray(bytearray(image_data), dtype="uint8")
            origin_frame = cv2.imdecode(image, cv2.IMREAD_COLOR)

            # read image as rgb
            origin_frame = origin_frame[:, :, ::-1]
            width = origin_frame.shape[1]
            height = origin_frame.shape[0]

            # crop bottom half of the picture
            bottom_half = origin_frame[height - self.result_shape[0]:height, 0:width]

            img = np.array(
                Image.fromarray(bottom_half.astype(np.uint8, copy=False)).resize(
                    (self.input_shape[1], self.input_shape[0]), Image.NEAREST))
            # img = np.array(Image.fromarray(bottom_half.astype(np.uint8, copy=False)))
            img = np.asarray(img).transpose([2, 0, 1]).astype(np.float32).copy(order='C')
            # img = img.transpose(2, 0, 1)
            # self.batch_data[0][0] = img
            # self.mod.forward(mx.io.DataBatch(data=self.batch_data, label=self.batch_label), is_train=False)
            #
            # if self._flipping:
            #     preds = copy.deepcopy(self.mod.get_outputs())
            #     flip_batch_data = []
            #     for batch_split_data in self.batch_data:
            #         flip_batch_data.append(mx.nd.array(batch_split_data.asnumpy()[:, :, :, ::-1]))
            #     self.mod.forward(mx.io.DataBatch(flip_batch_data, label=self.batch_label), is_train=False)
            #     flip_preds = self.mod.get_outputs()
            #     merge_preds = []
            #     for i, pred in enumerate(preds):
            #         # change left-lane and right-lane dimension when flipplig
            #         flipped_pred = flip_preds[i].asnumpy()[:, :, :, ::-1]
            #         flipped_pred[:, [1, 2], :, :] = flipped_pred[:, [2, 1], :, :]
            #         merge_preds.append(mx.nd.array((0.5 * pred.asnumpy() + 0.5 * flipped_pred)))
            #     preds = merge_preds
            # else:
            #     preds = self.mod.get_outputs()
            #
            # self.upsample_mod.forward(mx.io.DataBatch(data=preds), is_train=False)
            # out_pred = self.upsample_mod.get_outputs()[0].asnumpy().squeeze().astype(np.uint8)
            # self.batch_data[0] = img.astype(np.float32)
            # tic = time.time()
            out_pred = self.trt_infer.infer(img)[0]
            _time1 = time.time()
            rgb_frame = self.label_to_color[out_pred]
            _time2 = time.time()
            print("id to color use:{} s".format(_time2 - _time1))
            Image.fromarray(rgb_frame.reshape(512, 1224, 3).astype(np.uint8)).resize((2448, 1024),
                                                                                     Image.NEAREST).save(dest_file)
            # img_array = cv2.imencode('.png', rgb_frame)
            # img_data = img_array[1]
            # pred_data = img_data.tostring()
            #
            # if dest_file is not None:
            #     with open(dest_file, "wb") as f:
            #         f.write(pred_data)
            # # dest_path = os.path.join(os.path.dirname(dest_file), os.path.basename(dest_file)[:-4]+"-vir.png")
            # # cv2.imwrite(dest_path, rgb_frame)
            #
            # return pred_data
        except Exception as e:
            print("recognition error:{}".format(repr(e)))
        finally:
            return pred_data


class Task:

    def __init__(self, img_data, dest_path, exit_flag=False):
        self.img_data = img_data
        self.dest_path = dest_path
        self.exit_flag = exit_flag


def do_seg(gpu_id, task_queue, model_path, count, cls, seg_label_li, lock, input_size, output_size, argmax,
           effective_c):
    net = ModelResNetRoad(
        gpu_id=gpu_id,
        model_path=model_path,
        cls=cls,
        seg_label_li=seg_label_li,
        input_size=input_size,
        output_size=output_size,
        argmax=argmax,
        effective_c=effective_c)
    while True:
        if task_queue.empty():
            break

        task = task_queue.get()

        if not isinstance(task, Task):
            break

        if task.exit_flag:
            break

        img_data = task.img_data
        dest_path = task.dest_path

        try:
            start = time.time()
            net.do(image_data=img_data, dest_file=dest_path)
            lock.acquire()
            count.value += 1
            end = time.time()
            lock.release()
            print("Processed {} in {} ms".format(dest_path, str((end - start) * 1000)))
        except Exception as e:
            print(repr(e))
