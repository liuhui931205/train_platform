# -*- coding:utf-8 -*-
def get_line_count(filename):
    line_count = 0
    files = open(filename, 'r+')
    while True:
        buffers = files.read(8192 * 1024)
        if not buffers:
            break
        line_count += buffers.count('\n')
    files.close()
    return line_count
