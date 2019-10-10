# coding=utf-8
# 此蓝图用于提供静态页面
# from flask import Blueprint, current_app, make_response
#
# html = Blueprint('html', __name__)
#
#
# @html.route('/<re(".*"):file_name>')
# def send_html_file(file_name):
#     # 获取静态页面
#     if file_name == '':
#         file_name = 'index.html'
#     file_name = file_name
#     response = make_response(current_app.send_static_file(file_name))
#     return response
