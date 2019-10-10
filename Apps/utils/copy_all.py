# -*- coding:utf-8 -*-
import os


def copyFiles(sourceDir, targetDir):
    for f in os.listdir(sourceDir):
        sourceF = os.path.join(sourceDir, f)
        targetF = os.path.join(targetDir, f)
        if os.path.isfile(sourceF):
            # 创建目录
            if not os.path.exists(targetDir):
                os.makedirs(targetDir)
                # 文件不存在，或者存在但是大小不同，覆盖
            if not os.path.exists(targetF) or (
                    os.path.exists(targetF) and (os.path.getsize(targetF) != os.path.getsize(sourceF))):
                open(targetF, "wb").write(open(sourceF, "rb").read())
            else:
                pass
        elif os.path.isdir(sourceF):
            if not os.path.exists(targetF):
                os.makedirs(targetF)
            copyFiles(sourceF, targetF)
