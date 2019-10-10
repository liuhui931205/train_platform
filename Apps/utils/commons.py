# -*- coding:utf-8 -*-
from werkzeug.routing import BaseConverter


class RegexConverter(BaseConverter):
    def __init__(self, url_map, regex):
        super(RegexConverter, self).__init__(url_map)

        # 保存匹配规则的正则表达式
        self.regex = regex
