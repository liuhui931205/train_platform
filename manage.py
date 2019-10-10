# -*- coding:utf-8 -*-
from Apps import create_app, models

# app = create_app('development')
app = create_app('production')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9001)
    # app.run()
