# -*-coding:utf-8-*-
import subprocess
import argparse


def work(params,
         network,
         gpu,
         multiple4=0,
         shape="\'(1,3,512,1224)\'",
         maxBatchSize=1,
         iname="data",
         oname="prob",
         OIndex=0,
         topk=0,
         fp16=0,
         int8=0,
         imagelstInt8=0,
         reuseCacheInt8=0,
         batchSizeInt8=32,
         maxBatchesInt8=20):
    cmd = '/data/deeplearning/train_platform/tensorRT/model_Mxnet2Trt --network ' + network + ' --params ' + params + ' --multiple4 ' + str(
        multiple4) + ' --shape ' + shape + ' --gpu ' + str(gpu) + ' --maxBatchSize ' + str(
            maxBatchSize) + ' --iname ' + iname + ' --oname ' + oname + ' --OIndex ' + str(OIndex) + ' --topk ' + str(
                topk) + ' --fp16 ' + str(fp16) + ' --int8 ' + str(int8) + ' --imagelstInt8 ' + str(
                    imagelstInt8) + ' --reuseCacheInt8 ' + str(reuseCacheInt8) + ' --batchSizeInt8 ' + str(
                        batchSizeInt8) + ' --maxBatchesInt8 ' + str(maxBatchesInt8)
    print cmd
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    # print(p.stdout.read())
    cmd2 = '/data/deeplearning/train_platform/tensorRT/build_trt --network ' + network + ' --params ' + params + ' --multiple4 ' + str(
        multiple4) + ' --shape ' + shape + ' --gpu ' + str(gpu) + ' --maxBatchSize ' + str(
            maxBatchSize) + ' --iname ' + iname + ' --oname ' + oname + ' --OIndex ' + str(OIndex) + ' --topk ' + str(
                topk) + ' --fp16 ' + str(fp16) + ' --int8 ' + str(int8) + ' --imagelstInt8 ' + str(
                    imagelstInt8) + ' --reuseCacheInt8 ' + str(reuseCacheInt8) + ' --batchSizeInt8 ' + str(
                        batchSizeInt8) + ' --maxBatchesInt8 ' + str(maxBatchesInt8)
    s = p.communicate()
    if s:
        if s[0].split('\n')[-2].startswith('FINISH:'):
            q = subprocess.Popen(
                cmd2,
                shell=True,
                stdout=subprocess.PIPE)
            w = q.communicate()
            if w[0].split('\n')[-2].startswith('FINISH'):
                return "success"
            else:
                return "build_trt run error"
        else:
            return "model_Mxnet2Trt run error"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, required=True)
    parser.add_argument('--params', type=str, required=True)
    args = parser.parse_args()
    network = args.network
    params = args.params
    out = work(params, network)
    print out
