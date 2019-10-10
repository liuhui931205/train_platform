
//点击训练任务
$(".trainTask").click(function(){
	trainTask.Table();
})
//训练任务管理
var trainTask = {
	
	checkTask: {},
	
	dataLanguage: {
		"batch_size" : "批大小",
		"category_num" : "类别数目",
		"task_id" : "taskId",
		"machine_id" : "机器",
		"data_name": "数据名",
		"data_desc" : "data描述",
		"end_date" : "结束时间",
		"gpus" : "GPU",
		"id" : "id",
		"iter_num" : "全量迭代",
		"learning_rate" : "初始学习率",
		"net_desc" : "network描述",
		"start_date" : "开始时间",
		"steps" : "学习率调整步长",
		"task_desc" : "task描述",
		"task_name" : "任务名"
	},

//	模型的模版列表
	Table: function(){

//  	训练任务列表的模板
	    var listModel = ` 
	    <div class="row">
		    <div class='testmodal'></div>
        	<div class="col-xs-12">
        		<div class="box box-primary">
        			<div class="box-header">
        				<h3 class="box-title">训练任务</h3>
    				</div>
				    <div class="taskButtons" style="padding-left:10px;">
				      	<button type="button" onclick="createTask.createModel()" class="btn btn-success" data-toggle="modal" data-target="#modal-default">
				      		创建训练任务 / 优化训练任务
				      	</button>
				    </div>
	    			<div class="box-body">
						<table id="example2" class="table table-bordered table-hover">
					        <thead>
						        <tr>
									<th>选择</th>
									<th>id</th>
									<th>任务名</th>
									<th>机器</th>
									<th>taskId</th>
									<th>类别数目</th>
									<th>全量迭代</th>
									<th>初始学习率</th>
									<th>学习率调整步长</th>
									<th>批大小</th>
									<th>GPU</th>
									<th>开始时间</th>
									<th>结束时间</th>
									<th>操作</th>
						        </tr>
					        </thead>
		    			</table>
	    			</div>
	    		</div>
	    	</div>
	    </div>
	    `;

	    $('.content.container-fluid').html(listModel);

//		初始化表格(模型列表的属性)
    	var table = $("#example2").DataTable({
	    	'language'    : main.lang,
	        'searching'   : false,								//原生搜索
	        'paging'      : true,
	        'lengthChange': false,
	        'ordering'    : false,
	        'info'        : true,
	        'autoWidth'   : false,
	        "lengthMenu"  : [5],
	        "aaSorting"	  : false,
        	'processing'  : true,  								//隐藏加载提示,自行处理
        	'serverSide'  : false,  								//启用服务器端分页
	        ajax: function (data, callback, settings) {
	        	var param = {};
//				封装请求参数
	            param.limit = 5;								//页面显示记录条数，在页面显示每页显示多少项的时候
	            param.start = data.start;						//开始的记录序号
	            param.page = (data.start / data.length)+1;		//当前页码
				var url = trainUrl+"/train_history";
				var _data = {
					"taskId": ""
				}
//				使用post请求模型列表(每次只请求十条任务,然后对其进行分页)
	            $.ajax({
			        type : "post",
			        url : url,
			        async : true,
					contentType: "application/json; charset=utf-8",
			        data : JSON.stringify(_data),
	                success: function (data) {
	                	var taskData1 = data.data;
	                    var returnData = {};
	                    returnData.recordsTotal = taskData1.length;				//返回数据全部记录
	                    returnData.recordsFiltered = taskData1.length;			//后台不实现过滤功能，每次查询均视作全部结果
	                    returnData.data = taskData1;						//返回的数据列表
//						调用DataTables提供的callback方法，代表数据已封装完成并传回DataTables进行渲染
//						此时的数据需确保正确无误，异常判断应在执行此回调前自行处理完毕
                    	callback(returnData);
	                }
	           	});
	        },

//			模型列表的数据展示
        	columns: [
//      		第1列----选择----显示
		        {
	            	"data": "task_id",
	              	"render": function ( data, type, row,  meta ) {
	              		var id = row.id;
						var checkModel = `
							<label class="checkTask">
								<input type="checkbox" class="${id}" value="${data}" />
								<span></span>
							</label>
						`;
				    	return checkModel;
					},
					"class": "checkModel"
	           	},
//      		第2列----id----显示
		        {
	            	"data": "id",
					"class": "id"
	           	},
//      		第3列----任务名----显示
		        {
	            	"data": "task_name",
					"class": "task_name"
	           	},
//				第3列----机器----显示
		        {
	            	"data": "machine_id",
					"class": "machine_id"
	           	},
//      		第4列----taskId----显示
		        {
	            	"data": "task_id",
					"class": "task_id"
	           	},
//      		第5列----类别数目----显示
		        {
	            	"data": "category_num",
					"class": "category_num"
	           	},
//      		第6列----全量迭代----显示
		        {
	            	"data": "iter_num",
					"class": "iter_num"
	           	},
//      		第7列----初始学习率----显示
		        {
	            	"data": "learning_rate",
					"class": "learning_rate"
	           	},
//      		第8列----学习率调整步长----显示
		        {
	            	"data": "steps",
					"class": "steps"
	           	},
//      		第9列----批大小----显示
		        {
	            	"data": "batch_size",
					"class": "batch_size"
	           	},
//      		第10列----GPU----显示
		        {
	            	"data": "gpus",
					"class": "gpus"
	           	},
//      		第11列----开始时间----显示
		        {
	            	"data": "start_date",
					"class": "start_date"
	           	},
//      		第12列----结束时间----显示
		        {
	            	"data": "end_date",
					"class": "end_date"
	           	},
//      		第13列----操作按钮----显示(预览/删除/清除)
	            {
	            	"data": "id",
	              	"render": function ( data, type, row,  meta ) {
	              		var taskId = row.task_id;
						var centerModel = `
							<div class="btn-group">
								<button type="button" class="btn btn-warning details" title="${taskId}" data-toggle="modal" data-target="#modal-default" onclick="trainTask.details('${taskId}')">详情</button>
								<button type="button" class="btn btn-warning logs" title="${taskId}" data-toggle="modal" data-target="#modal-default" onclick="trainTask.logs('${taskId}')">日志</button>
								<button type="button" class="btn btn-warning Continue" title="${taskId}" onclick="trainTask.Continue('${taskId}')">继续训练</button>
		                    </div>
						`;
				    	return centerModel;
					},
					"class": "center"
	            }
	        ]
	    });

	    $('#example2 tbody').on('click', 'tr td .checkTask input', function () {
	    	if(this.checked){
	    		trainTask.checkTask[this.className] = this.value;
	    	}else{
	    		trainTask.checkTask[this.className] = "";
	    	}
	    });

//		列表点击分页时，之前选中的任务进行再次选中
	    $('#example2_paginate').on('click', function () {
			for(var item in trainTask.checkTask){
				if( trainTask.checkTask[item] && $('.'+item).length ){
					$('.'+item)[0].checked = true;
				}
			}
	    });
	},

	detailsData: function(id){
		var datas = {};
		var form = {
			"task_id" : id
		};
		var urls = {
			"describe" : trainUrl+"/train_history",
			"train_configs" : trainUrl+"/train_configs?task_id="+id,
			"data_configs" : trainUrl+"/data_configs?task_id="+id,
			"map_info" : trainUrl+"/map_info?task_id="+id
		};
		for(var item in urls){
			if(item == "describe"){
				var data = util.newAjax(urls[item],"POST",form);
				datas[item] = data;
			}else{
				var data = util.newAjax(urls[item],"GET");
				datas[item] = data;
			}
		}
		return datas;
	},

//	操作按钮继续训练的功能按钮
	Continue: function(taskId){
		var ContinueUrl = trainUrl+"/continue_task";
		var parms = {
			"task_id": taskId
		};
		util.newAjax(ContinueUrl,"POST",parms,function(result){
			if( result.errno == 1 ){
				util.errorView(result.message);
			}else{
				util.errorView(result.message);
			}
			return;
		});
	},

//	查看详情
	details: function(id){
		var datas = trainTask.detailsData(id);
		var taskView = [],
			map_infos = datas["map_info"].data;

		for(var item in datas["describe"].data[0]){
			var name = item,
				value = datas["describe"].data[0][item];
			if(trainTask.dataLanguage[name]){
				taskView.push({
					"name" : trainTask.dataLanguage[name],
					"value" : value
				})
			}
		}
		var tempModel =`
			<div class="modal fade task-modal" data-backdrop="static" data-show="true" id="modal-default">
				<div class="modal-dialog task-dialog" style="width:1000px;">
					<div class="modal-content task-modal-content">
						<div class="modal-header" style="height:50px;">
							<button type="button" class="close" title="关闭" onclick="main.modalClose()">
								<span aria-hidden="true" class="glyphicon glyphicon-remove"></span>
							</button>
							<h4 class="modal-title">任务详情 (${id})</h4>
						</div>
						
						<div class="modal-body task-modal-body" style="height:calc(100% - 100px) !important;max-height:900px;overflow-x:auto;">
							<div class="task-top">
				                ${taskView.map(t => `
									<p>${t.name}: ${t.value || '无'}</p>
				                `).join('')}
							</div>
							<div class="task-bottom">
								<div>
									<h4>训练配置</h4>
			    					<textarea class="result1" readonly="readonly"></textarea>
								</div>
								<div style="float:right;">
									<h4>数据配置</h4>
				    				<textarea class="result2" readonly="readonly"></textarea>
								</div>
							</div>
							<div>
								<div>
									<h4>类别映射</h4>
									<table id="exampleInfo" class="table table-bordered table-hover">
							            <thead style="font-weight:bold;">
							                <tr>
												<th>id</th>
												<th>categoryId</th>
												<th>en_name</th>
												<th>name</th>
												<th>color</th>
							                </tr>
							            </thead>
								        <tbody>
							                ${map_infos.map(m => `
									            <tr>
										            <td title="${m.id}">${m.id}</td>
										            <td title="${m.categoryId}">${m.categoryId}</td>
										            <td title="${m.en_name}">${m.en_name}</td>
										            <td title="${m.name}">${m.name}</td>
                                                    <td title="${m.color}" style="background:rgb(${m.color})"></td>
									            </tr>
							                `).join('')}
								        </tbody>
									</table>
								</div>
							</div>
						</div>
					</div>
				</div>
			</div>
		`;
		$('.testmodal').html(tempModel);
//	    $('#modal-default').modal('show');
		$('.result1').text( util.formatJson(datas["train_configs"].data) );
		$('.result2').text( util.formatJson(datas["data_configs"].data) );
//		table列表的属性样式等
    	$('#exampleInfo').DataTable({
			'language'    : window.lang,
            'paging'      : true,
            'lengthChange': false,
            'searching'   : false,
            'ordering'    : false,
            'info'        : true,
            'autoWidth'   : false,
            "lengthMenu"  : [10]
    	})
	},

//	查看日志
	logs: function(id){
//		监控界面的模板
	    var templ =`
			<div class="modal fade task-modal" data-backdrop="static" data-show="true" id="modal-default">
				<div class="modal-dialog task-dialog" style="width:1000px;">
					<div class="modal-content task-modal-content">
						<div class="modal-header" style="height:50px;">
							<button type="button" class="close" title="关闭" onclick="main.modalClose()">
								<span aria-hidden="true" class="glyphicon glyphicon-remove"></span>
							</button>
							<h4 class="modal-title">任务日志 (${id})</h4>
						</div>
						
	              		<div class="modal-body flowing-modal-body">
		              		<!-- 监控界面 -->
							<div style="height:100%;">
		              			<!-- websocket日志界面 -->
							    <div id="log-container"></div>
							</div>
	              		</div>
					</div>
				</div>
			</div>
	    `;
	    $('.testmodal').html(templ);
//	    $('#modal-default').modal('show');
		var logUrl = trainUrl+"/output_log?task_id="+id;
		taskLog.logTime = setInterval(function(){
			util.newAjax(logUrl,"GET","",function(result){
				$("#log-container").html("");
				if( result.errno == 1 ){
	            	for(var i=0; i<result.data.length; i++){
	            		if(result.data[i]){
	            			var timeText = `<span>${result.data[i]}</span>`;
							$("#log-container").append(timeText);
							$("#log-container").scrollTop( $('#log-container')[0].scrollHeight );
	            		}
	            	}
				}else{
					util.errorView(result.message);
					return;
				}
			});
		},10000);
	},
}
//初始化显示模版
trainTask.Table();
