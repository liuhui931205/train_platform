# 接口文档
### 一、训练部分
#### 1、查询历史任务
请求URL：/train_history

---

请求方式：post

---

请求参数：

参数 | 实例|说明
---|---|---
无 | 无|无

---
返回数据：

key   |   value  | 说明
--------|--------|-----
errno | 1 | 状态码
data    |   record | 所有数据（id（ID）,task_id（任务id）,task_name（任务名）,network_path(network路径),data_path（data路径）,category_num（类别数目）,iter_num（全量迭代）,learning_rate（初始学习率）,steps（学习率调整步长）,batch_size（批大小）,gpus（GPU）,start_date（开始时间）,end_date（结束时间）,net_desc（network描述），task_desc（任务描述），data_desc（数据描述），status（状态））

---

---

#### 2、查询任务详情
请求URL：/train_history

---

请求方式：post

---

请求参数：

参数 | 实例|说明
---|---|---
task_id | 'task_id': 'sasa'|任务ID,str

---
返回数据：

key   |   value  | 说明
--------|--------|-----
errno | '1' | 状态码
data    |   desc | 任务详情（id（ID）,task_id（任务id）,task_name（任务名）,network_path(network路径),data_path（data路径）,category_num（类别数目）,iter_num（全量迭代）,learning_rate（初始学习率）,steps（学习率调整步长）,batch_size（批大小）,gpus（GPU）,start_date（开始时间）,end_date（结束时间）,net_desc（network描述），task_desc（任务描述），data_desc（数据描述），status（状态））

---

---
#### 3、查询训练配置
请求URL：/train_configs

---

请求方式：get

---

请求参数：

参数 | 实例|说明
---|---|---
task_id | 'task_id': 'dsad'|任务ID,str

---
返回数据：

key   |   value  | 说明
--------|--------|-----
data    |   resp_data | 读取json文件的数据
errno    |   1 | 状态码
---

---
#### 4、查询数据配置
请求URL：/data_configs

---

请求方式：get

---

请求参数：

参数 | 实例|说明
---|---|---
task_id | 'task_id': 'sdas'|任务ID,str

---
返回数据：

key   |   value  | 说明
--------|--------|-----
resp    |   resp_data | 读取json文件的数据
errno    |   1 | 状态码

---

---
#### 5、创建训练任务
请求URL：/create_train

---

请求方式：post

---

请求参数：

参数 | 实例|说明
---|---|---
train_name | 'train_name': 'suzhou'|输入任务名（字符串）
network | 'network': 'c_network'|选择的network文件夹名称（可选）
data | 'data': '20180727'|选择的data文件夹名称（可选）
task_desc | 'task_desc': '苏州任务'|输入任务描述（可输入中文）

---
返回数据：

key   |   value  | 说明
--------|--------|-----
message    |   'Autosele start' | 
errno    |  1 | 状态码（成功）
data    |  task_id | 任务id

---

---
#### 5、获取network文件夹名
请求URL：/network

---

请求方式：get

---

请求参数：

参数 | 实例|说明
---|---|---
 无|无 |无


---
返回数据：

key   |   value  | 说明
--------|--------|-----
data    |   network_li | network文件夹名称列表
errno    |  1 | 状态码（成功）

---

---
#### 6、获取data文件夹名
请求URL：/sour_datas

---

请求方式：get

---

请求参数：

参数 | 实例|说明
---|---|---
 无|无 |无


---
返回数据：

key   |   value  | 说明
--------|--------|-----
data    |   datas_li | data文件夹名称列表
errno    |   1 | 状态码（成功）

---

---
#### 7、获取label_map映射
请求URL：/maps

---

请求方式：post

---

请求参数：

参数 | 实例|说明
---|---|---
 task_id|'task_id':'sadasd' |任务id,str


---
返回数据：

key   |   value  | 说明
--------|--------|-----
data    |   'id_list' | {0:['12','other','其他'],1:['255','ignore','Ignore']...}
errno    |   1 | 状态码（成功）

---

---
#### 8、保存label_map映射
请求URL：/save_map

---

请求方式：post

---

请求参数：

参数 | 实例|说明
---|---|---
 map_value|'map_value':{'0':['other','其他','12'],...} |字典
  task_id|'task_id':'sadasd' |任务id,str


---
返回数据：

key   |   value  | 说明
--------|--------|-----
message    |   'save success' | 
errno    |   1 | 状态码（成功）

---

---
#### 9、选择weights
请求URL：/weights

---

请求方式：get

---

请求参数：

参数 | 实例|说明
---|---|---
task_id|'task_id':'sadasd' |任务id,str


---
返回数据：

key   |   value  | 说明
--------|-------|-----
data    |   li | weights文件夹里文件的列表
message    |   'query success' | 
errno    |   1 | 状态码（成功）


---

---
#### 10、查询修改训练配置
请求URL：/train_config

---

请求方式：get

---

请求参数：

参数 | 实例|说明
---|---|---
task_id|'task_id':'sadasd' |任务id,str


---
返回数据：

key   |   value  | 说明
--------|--------|-----
data    |   'resp_data' | 读取并修改训练配置的json文件
message    |   'query success' | 
errno    |   1 | 状态码（成功）


---

---
#### 11、查询修改数据配置
请求URL：/data_config

---

请求方式：get

---

请求参数：

参数 | 实例|说明
---|---|---
task_id|'task_id':'sadasd' |任务id,str


---
返回数据：

key   |   value  | 说明
--------|--------|-----
data    |   'resp_data' | 读取并修改数据配置的json文件
message    |   'query success' | 
errno    |   1 | 状态码（成功）

---

---
#### 12、参数自动计算
请求URL：/calculate

---

请求方式：get

---

请求参数：

参数 | 实例|说明
---|---|---
 batch_val|24 |批大小的值（整数）
 task_id|'task_id':'sadasd' |任务id,str


---
返回数据：

key   |   value  | 说明
--------|--------|-----
data    |   're_val' | 全量迭代的值
message    |   'success' | 全量迭代的值
errno    |   1 | 状态码（成功）

---

---
#### 13、开始训练任务
请求URL：/starttask

---

请求方式：post

---

请求参数：

参数 | 实例|说明
---|---|---
task_id|'task_id':'asdasd' |任务id
data_type|'data_type':'all' |数据类型（可选）
num_classes|'num_classes':'13' |类别数目（整数）
num_epoch|'num_epoch':'13' |类别数目（整数）
batch_size|'batch_size':'24' |批大小的值（整数）
gpus|'gpus':'0,1,2' |GPU的id以逗号分开（字符串）
model|'model':'kd_suzhou' |模型前缀 （字符串）
steps|'steps':'10000,20000,30000' |学习率调整步长（字符串）
base_lr|'base_lr':'0.014' |初始学习率（浮点数）
start_date|'start_date':'2018-07-31 12:22:30' |开始时间（时间）
train_con|'train_con':{'ss':sdas} |训练配置（json数据）
data_con|'data_con':{'ss':sdas} |数据配置（json数据）
weights|'weights':'cityscapes_rna-a1_cls19_s8_ep-0001.params' |weights文件名（可选）



---
返回数据：

key   |   value  | 说明
--------|--------|-----
message    |   '任务开始' | 
errno    |   1 | 状态码（成功）

---

---
#### 14、结束训练任务
请求URL：/endtask

---

请求方式：post

---

请求参数：

参数 | 实例|说明
---|---|---
end_date|'end_date':'2018-07-31 12:22:30' |结束时间（时间）
task_id|'task_id':'asdasd' |任务id




---
返回数据：

key   |   value  | 说明
--------|--------|-----
message    |   '任务结束' | 
errno    |   '1' | 状态码（成功）

---

---
#### 15、查询已存在network
请求URL：/network_history

---

请求方式：get

---

请求参数：

参数 | 实例|说明
---|---|---
无|无 |无

---
返回数据：

key   |   value  | 说明
--------|--------|-----
errno | '1' | 状态码
data    |   record | （id（ID）,task_id（任务id），status(状态),net_name（network名）,net_describe（network描述））



---

---
#### 16、选择network源文件夹
请求URL：/src_network

---

请求方式：get

---

请求参数：

参数 | 实例|说明
---|---|---
无 | 无|无

---
返回数据：

key   |   value  | 说明
--------|--------|-----
errno | '1' | 状态码
data    |   src_network_li | network源文件夹的名称列表

---

---
#### 17、创建network
请求URL：/create_net

---

请求方式：post

---

请求参数：

参数 | 实例|说明
---|---|---
network_name | 'network_name':'c_network'|network名称（字符串）
src_net | 'src_net':'kd-seg.template'|network源文件夹名称（可选）
net_describe | 'netwonet_describe':'神经网络'|network描述（支持中文）

---
返回数据：

key   |   value  | 说明
--------|--------|-----
errno |1 | 状态码
message    |   '创建成功' | 
data    |   task_id |任务id
---

---
#### 18、查询已存在data
请求URL：/data_history

---

请求方式：POST

---

请求参数：

参数 | 实例|说明
---|---|---
无 |无|无

---
返回数据：

key   |   value  | 说明
--------|--------|-----
errno | '1' | 状态码
data    |   record | （id（ID）,task_id（任务id）,data_name（data名）,data_describe（data描述）,data_type(任务类型)，status(状态)，sour_data（源数据），train，val，test）
---

---
#### 18、生成data进度
请求URL：/data_history

---

请求方式：POST

---

请求参数：

参数 | 实例|说明
---|---|---
task_id |'task_id':'sdasa'|任务id

---
返回数据：

key   |   value  | 说明
--------|--------|-----
errno | '1' | 状态码
data    |   record | （id（ID）,task_id（任务id）,data_name（data名）,data_describe（data描述）,data_type(任务类型)，status(状态)，sour_data（源数据），train，val，test）

---

---
#### 19、选择data源文件夹
请求URL：/data

---

请求方式：get

---

请求参数：

参数 | 实例|说明
---|---|---
无 | 无|无

---
返回数据：

key   |   value  | 说明
--------|--------|-----
errno | 1 | 状态码
data    |   li | data源文件夹的名称列表

---

---
#### 20、生成data
请求URL：/create_data

---

请求方式：post

---

请求参数：

参数 | 实例|说明
---|---|---
data_name | 'data_name':'20180801'|data名称（字符串）
data_type | 'data_type':'all'|数据类型（可选）
sour_data | 'sour_data':'lane-all-20180710'|data源文件夹名称（可选）
thread | 'thread':'24'|进程数（整数）
train | 'train':'0.85'|（浮点数）
val | 'val':'0.1'|（浮点数）
test | 'test':'0.1'|（浮点数）
data_desc | 'data_desc':'苏州数据'|数据描述（支持中文）
l_value | 'l_value':'1'|bool值


---
返回数据：

key   |   value  | 说明
--------|--------|-----
errno | 1 | 状态码
message    |   'data已生成' | 
data    |   'task_id' | 任务id

### 二、模型部分
#### 1、查询生成的模型
请求URL：/model

---

请求方式：get

---

请求参数：

参数 | 实例|说明
---|---|---
无 | 无|无

---
返回数据：

key   |   value  | 说明
--------|--------|-----
errno | 1 | 状态码
data    |   data | 包含任务名task_name和任务的模型文件名


### 三、评估部分
#### 1、开始评估
请求URL：/starteva

---

请求方式：post

---

请求参数：

参数 | 实例|说明
---|---|---
dicts | 'dicts': {'suzhou':['kd_suzhou_test02_ep_0205.params']}|参数形式为字典，key为任务名，value为选择的模型名
type | 'type': 'remote'|评估类型
sour_dir | 'sour_dir': 'image_dir'|需要评估的图片地址（字符串）
dest_dir | 'dest_dir': 'label_dir'|已评估的图片存放地址（字符串）
gpus | 'gpus': '0,1,2'|以逗号分隔（字符串）
single_gpu | 'single_gpu': '2'|单个gpu处理图片的数目（整数）

---
返回数据：

key   |   value  | 说明
--------|--------|-----
errno | 1 | 状态码
message | '模型预测完成' | 
data | 'task_id' | 任务id
---

---
#### 2、评估进度(重复请求)
请求URL：/eva_rate

---

请求方式：POST

---

请求参数：

参数 | 实例|说明
---|---|---
task_id | 'task_id': 'sadsa'|任务id


---
返回数据：

key   |   value  | 说明
--------|--------|-----
errno | 1 | 状态码
data | 任务进度 |（task_id，id,type,sour_dir,gpus,dest_dir,single_gpu,model,status）
---

---
#### 3、展示评估后的数据
请求URL：/show

---

请求方式：POST

---

请求参数：

参数 | 实例|说明
---|---|---
cur_img | 'cur_img': 1|页数（整数）
task_id | 'task_id': 'sadsa'|任务id


---
返回数据：

key   |   value  | 说明
--------|--------|-----
errno | 1 | 状态码
or_data | or_data |原始图片数据 
la_data | la_data |预测图片数据 
total_img | total_img |总页数



### 四、发布部分
#### 1、已存在模型展示
请求URL：/release_tab

---

请求方式：get

---

请求参数：

参数 | 实例|说明
---|---|---
page | 'page': 1|查询页数（整数）

---
返回数据：

key   |   value  | 说明
--------|--------|-----
errno | '1' | 状态码
data | 'data' | 模型文件夹名和模型文件名
name | '车道线' | 类别名
total | '3' | 总页数
---

---
#### 3、开始发布模型
请求URL：/release

---

请求方式：post

---

请求参数：

参数 | 实例|说明
---|---|---
dicts | 'dicts': {'20180427':[kdss.params,kdss.json]}|必须选择以json和params结尾的文件（字典）
page | 'page': '1'|查询页数（整数）
version | 'version': '20180731'|版本号（字符串）
env | 'env': '北京'|环境（字符串）
adcode | 'adcode': '1110000'|（整数）
desc | 'desc': '北京模型'|描述（支持中文）
type | 'type': '车道线'|模型类型（可选）
time | 'time': '2018-07-31 12:22:30'|发布时间（时间）

---
返回数据：

key   |   value  | 说明
--------|--------|-----
errno | 1 | 状态码
data | 'resp' | 发布后返回的信息

### 五、筛选数据
#### 1、日志
请求URL：/output_log 

---

请求方式：get

---

请求参数：

参数 | 实例|说明
---|---|---
task_id | 'task_id': 'sadsa'|任务id
---
返回数据：

key   |   value  | 说明
--------|--------|-----
errno | 1 | 状态码
data | data | 日志数据

### 六、筛选数据
#### 1、随机抽图
请求URL：/sam_value

---

请求方式：post

---

请求参数：

参数 | 实例|说明
---|---|---
output_dir | 'output_dir': '/data/'|输出地址
ratio | 'ratio': '1'|筛选张数
track_file | 'track_file': '/data'|track_file地址
task_file | 'task_file': '/data'|task_file地址（track_file地址与task_file地址输入一个即可）
isshuffle | 'isshuffle': '1'|是否打乱顺序


---
返回数据：

key   |   value  | 说明
--------|--------|-----
errno | 1 | 状态码
message | 'success' | 返回的信息
data | task_id | 任务id
---

---
#### 2、随机抽图进度（重复请求）
请求URL：/auto_sam

---

请求方式：post

---

请求参数：

参数 | 实例|说明
---|---|---
task_id | 'task_id': 'sadasd'|任务id

---
返回数据：

key   |   value  | 说明
--------|--------|-----
errno | 1 | 状态码
data | 任务详情 | （task_id，task_type，output_dir，gpus，sele_ratio，weights_dir，track_file，task_file，isshuffle，status）
---
---
#### 3、自动挑图
请求URL：/sele_value

---

请求方式：post

---

请求参数：

参数 | 实例|说明
---|---|---
output_dir | 'output_dir': '/data/'|输出地址
ratio | 'ratio': '1'|筛选张数
gpus| 'gpus': '1,2'|gpu
weights_dir| 'weights_dir': '/data/'|模型地址
track_file | 'track_file': '/data'|track_file地址
isshuffle | 'isshuffle': '1'|是否打乱顺序


---
返回数据：

key   |   value  | 说明
--------|--------|-----
errno | 1 | 状态码
message | 'success' | 返回的信息
data | task_id | 任务id
---

---
#### 4、自动挑图进度（重复请求）
请求URL：/auto_sele

---

请求方式：post

---

请求参数：

参数 | 实例|说明
---|---|---
task_id | 'task_id': 'sadasd'|任务id

---
返回数据：

key   |   value  | 说明
--------|--------|-----
errno | 1 | 状态码
data | 任务详情 | （task_id，task_type，output_dir，gpus，sele_ratio，weights_dir，track_file，task_file，isshuffle，status）


### 七、GPU
#### 1、gpu
请求URL：/gups_info

---

请求方式：get

---

请求参数：

参数 | 实例|说明
---|---|---
无 | |


---
返回数据：

key   |   value  | 说明
--------|--------|-----
errno | 1 | 状态码
data | gpu_li | gpu信息列表


### 八、数据
#### 1、离线导入
请求URL：/offimport

---

请求方式：post

---

请求参数：

参数 | 实例|说明
---|---|---
roadelement |"roadelement":"sad" |色板信息
source |"source":"sad" |数据来源
author |"author":"sad" |作者信息
annotype | "annotype":"sad"|标注类型
datakind |"datakind":"sad" |数据种类
city |"city":"sad" |城市
src |"src":"sad" |输入
dest | "dest":"sad"|输出位置
imgoprange | "imgoprange":"sad"|图片操作范围
---
返回数据：

key   |   value  | 说明
--------|--------|-----
errno | 1 | 状态码
message | 'success' | 成功

#### 2、离线导入进度（重复请求）
请求URL：/off_status

---

请求方式：post

---

请求参数：

参数 | 实例|说明
---|---|---
task_id | 'task_id': 'sadasd'|任务id

---
返回数据：

key   |   value  | 说明
--------|--------|-----
errno | 1 | 状态码
data | 任务详情 | （task_id，roadelement，source，author，annotype，datakind，city，imgoprange，status）

#### 3、在线下载
请求URL：/linedown

---

请求方式：post

---

请求参数：

参数 | 实例|说明
---|---|---
taskid_start |"taskid_start":"11" |起始taskid
taskid_end |"taskid_end":"sad" |结束taskid
dest |"dest":"asdsa" |输出位置
---
返回数据：

key   |   value  | 说明
--------|--------|-----
errno | 1 | 状态码
message | 'success' | 成功

#### 4、在线下载进度（重复请求）
请求URL：/line_status

---

请求方式：post

---

请求参数：

参数 | 实例|说明
---|---|---
task_id | 'task_id': 'sadasd'|任务id

---
返回数据：

key   |   value  | 说明
--------|--------|-----
errno | 1 | 状态码
data | 任务详情 | （task_id，taskid_start，taskid_end，dest，status）

#### 5、任务包生成
请求URL：/task_divide

---

请求方式：post

---

请求参数：

参数 | 实例|说明
---|---|---
version |"version":"dsad" |文件夹名
step |"step":"5" |
types |"types":"full" |full 或 remote 类型
---
返回数据：

key   |   value  | 说明
--------|--------|-----
errno | 1 | 状态码
message | 'success' | 成功

#### 6、任务包生成进度（重复请求）
请求URL：/divide_history

---

请求方式：post

---

请求参数：

参数 | 实例|说明
---|---|---
task_id | 'task_id': 'sadasd'|任务id

---
返回数据：

key   |   value  | 说明
--------|--------|-----
errno | 1 | 状态码
data | 任务详情 | （task_id，version，step，types，status）

#### 7、任务包发布
请求URL：/process_label

---

请求方式：post

---

请求参数：

参数 | 实例|说明
---|---|---
version |"version":"dasda" |文件夹名
types |"types":"remote" |remote、lane、union

---
返回数据：

key   |   value  | 说明
--------|--------|-----
errno | 1 | 状态码
message | 'success' | 成功

#### 8、任务包发布进度（重复请求）
请求URL：/label_history

---

请求方式：post

---

请求参数：

参数 | 实例|说明
---|---|---
task_id | 'task_id': 'sadasd'|任务id

---
返回数据：

key   |   value  | 说明
--------|--------|-----
errno | 1 | 状态码
data | 任务详情 | （task_id，version，types，name，status）