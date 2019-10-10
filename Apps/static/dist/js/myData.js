
//点击Data
$(".myData").click(function(){
	myData.Table();
})
//Data管理
var myData = {
//	Data管理的模版列表
	Table: function(){
		
//  	Data任务列表的模板
	    var listModel = ` 
	    <div class="row">
		    <div class='testmodal'></div>
        	<div class="col-xs-12">
        		<div class="box box-primary">
        			<div class="box-header">
        				<div class="buttons"></div>
        				<h3 class="box-title">Data</h3>
    				</div>
				    <div class="taskButtons" style="padding-left:10px;">
				      	<button type="button" onclick="myData.createModel()" class="btn btn-success">
				      		生成Data
				      	</button>
				    </div>
	    			<div class="box-body">
						<table id="example2" class="table table-bordered table-hover">
					        <thead>
						        <tr>
						            <th>选择</th>
					                <th>ID</th>
					                <th>taskId</th>
					                <th>Data名称</th>
					                <th>机器</th>
					                <th>原数据</th>
					                <th>Train</th>
					                <th>Val</th>
					                <th>Test</th>
					                <th>描述</th>
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

//		初始化表格(Data任务列表的属性)
    	var table = $("#example2").DataTable({
	    	'language'    : main.lang,
	        'searching'   : false,								//原生搜索
	        'paging'      : true,
	        'lengthChange': false,
	        'ordering'    : false,
	        'info'        : true,
	        'autoWidth'   : false,
	        "aaSorting"	  : false,
        	'processing'  : true,  								//隐藏加载提示,自行处理
        	'serverSide'  : false,  								//启用服务器端分页
	        ajax: function (data, callback, settings) {
	        	var param = {};
//				封装请求参数
	            param.limit = 10;								//页面显示记录条数，在页面显示每页显示多少项的时候
	            param.start = data.start;						//开始的记录序号
	            param.page = (data.start / data.length)+1;		//当前页码
				var params = {
					"task_id": ""
				}
//				使用post请求我的任务列表(每次只请求十条任务,然后对其进行分页)
	            $.ajax({
			        type : "post",
			        url : trainUrl+"/data_history",
			        async : false,
					contentType: "application/json; charset=utf-8",
			        data : JSON.stringify(params),
	                success: function (data) {
	                	taskData1 = data.data;
	                    var returnData = {};
	                    returnData.recordsTotal = data.data.length;				//返回数据全部记录
	                    returnData.recordsFiltered = data.data.length;			//后台不实现过滤功能，每次查询均视作全部结果
	                    returnData.data = data.data;						//返回的数据列表
//						调用DataTables提供的callback方法，代表数据已封装完成并传回DataTables进行渲染
//						此时的数据需确保正确无误，异常判断应在执行此回调前自行处理完毕
                    	callback(returnData);
	                }
	           	});
	        },
//			Data任务列表的数据展示
        	columns: [
//              第1列----选择----显示
		        {
	            	"data": "task_id",
	              	"render": function ( data, type, row,  meta ) {
	              		var id = row.id;
						var checkModel = `
							<label class="checkTask">
								<input type="radio" class="${id}" value="${data}" name="data"/>
								<span></span>
							</label>
						`;
				    	return checkModel;
					},
					"class": "checkModel"
	           	},
//      		第一列----ID----显示
	            {
	            	"data": "id" || "",
					"class": "id"
				},
//      		第二列----task_id----显示
	            {
	            	"data": "task_id" || "",
					"class": "task_id"
				},
//      		第三列----data名称----显示
	            {
	            	"data": "data_name" || "",
					"class": "data_name"
				},
//				第三列----机器----显示
	            {
	            	"data": "machine_id" || "",
					"class": "machine_id"
				},
//      		第四列----原数据----显示
	            { 
	            	"data": "sour_data" || "",
					"class": "sour_data"
				},
//      		第五列----train----显示
	            { 
	            	"data": "train" || "",
					"class": "train"
				},
//      		第六列----val----显示
	            { 
	            	"data": "val" || "",
					"class": "val"
				},
//      		第七列----test----显示
	            { 
	            	"data": "test" || "",
					"class": "test"
				},
//      		第八列----描述----显示
	            { 
	            	"data": "data_describe" || "",
					"class": "data_describe"
				},
//				第9列----操作按钮----显示详情
	            {
	            	"data": "task_id",
	              	"render": function ( data, type, row,  meta ) {
						var centerModel = `
							<div class="btn-group">
								<button type="button" class="btn btn-warning details" title="数据类别占比" data-toggle="modal" data-target="#modal-default" onclick="myData.details('${data}')">详情</button>
		                    </div>
						`;
				    	return centerModel;
					},
					"class": "center"
	            }
	        ]

		});

	    $('#example2 tbody').on('click', 'tr td .checkTask input', function () {
	    	trainTask.checkTask = {};
	    	trainTask.checkTask[this.className] = this.value;
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
	
//	生成Data
	createModel: function(){
		var checkArr = [];
		for(var item in trainTask.checkTask){
			if(trainTask.checkTask[item]){
				checkArr.push( trainTask.checkTask[item] );
			}
		}
		if(checkArr.length == 1){
			util.errorView("已选择1条Data");
		}
		var newData = [
			{
				"name":"Data名称",
				"text":"Data名称",
				"id":"data_name"
			},
			{
				"name":"数据类型",
				"id":"data_type",
				"type":"select",
				"data":[
					{
						"name":"all",
						"value":"all"
					},
					{
						"name":"aug",
						"value":"aug"
					}
				]
			},
			{
				"name":"图片类型",
				"id":"images_type",
				"type":"select",
				"data":[
					{
						"name":"full",
						"value":"full"
					},
					{
						"name":"half",
						"value":"half"
					}
				]
			},
			{
				"name":"原数据",
				"id":"sour_data",
				"type":"select",
				"url":trainUrl+"/data"
			},
			{
				"name":"类型",
				"id":"types",
				"type":"select",
				"url":trainUrl+"/dtask_type"
			},
			{
				"name":"线程数",
				"id":"thread",
				"value":"24"
			},
			{
				"name":"Train",
				"id":"train",
				"value":"0.85"
			},
			{
				"name":"Val",
				"id":"val",
				"value":"0.1"
			},
			{
				"name":"Test",
				"id":"test",
				"value":"0.05"
			},
			{
				"name":"是否生成数据列表文件",
				"id":"l_value",
				"type":"checkbox"
			},
			{
				"name":"Data描述",
				"id":"data_desc",
				"type":"text"
			},
		]
		var data = myNetwork.format(newData);
	    var templ =`
			<div class="modal fade task-modal" id="modal-default">
				<div class="modal-dialog ftp-dialog">
					<div class="modal-content task-modal-content">
						<div class="modal-header">
		    				<button type="button" class="close" title="关闭" onclick="main.modalClose()">
		    					<span aria-hidden="true">&times;</span>
		    				</button>
							<h4 class="modal-title">生成Data</h4>
						</div>
						<div class="modal-body task-modal-body">
						    <div class="right-content" style="margin:0 auto;width:350px;">
								<form id='taskForm2' class="form-horizontal">
			      					<div class="box-body">
										<!--遍历循环步骤一信息-->
						                ${data.map(d => `
											<div class="form-group">
												${d.html}
											</div>
						                `).join('')}
				      					<!-- 确定与取消按钮 -->
				      					<div class="box-footer">
							                <button type="button" class="btn btn-default" title="取消" onclick="main.modalClose()">取消</button>
							                <button type="button" class="btn btn-info pull-right" title="生成Data">生成Data</button>
				      					</div>
			          				</div>
			    				</form>
			  				</div>
						</div>
					</div>
				</div>
			</div>
	    `;
	    $('.testmodal').html(templ);
	    $('#modal-default').modal('show');
	    
	    $(".pull-right").click(function(){
	    	myData.submitForm(newData,checkArr)
	    })
	},
	
//	提交创建Data
	submitForm: function(newData,checkArr){
		var jsonObj = {};
		if(checkArr && checkArr[0]){
			jsonObj['task_id'] = checkArr[0];
		}
		for(var i=0; i<newData.length; i++){
			var id = newData[i].id;
			var value = $("#"+id).val() || $("#"+id+" option:selected").val() || "";
			if(newData[i].type == "checkbox"){
				value = $("#"+id)[0].checked || "";
			}
			jsonObj[id] = value;
		}
		var url = trainUrl+"/create_data";
		var processUrl = trainUrl+"/data_history";
		util.newAjax(url,"POST",jsonObj,function(result){
			if( result.errno == 1 ){
				main.modalClose();
				util.errorView(result.message,true);
				myData.Table();
				createTask.taskProgress(result.data.task_id,processUrl);
			}else{
				console.log(result);
				util.errorView(result.message);
		        return;
			}
		});
	},

//	查询数据类别占比
	details: function(taskId){
		var searchUrl = trainUrl+"/class_count?task_id="+taskId;
		var _data = util.newAjax(searchUrl,"GET");
		var tempModel =`
			<div class="modal fade task-modal" data-backdrop="static" data-show="true" id="modal-default">
				<div class="modal-dialog task-dialog" style="width:1000px;">
					<div class="modal-content task-modal-content">
						<div class="modal-header" style="height:50px;">
							<button type="button" class="close" title="关闭" onclick="main.modalClose()">
								<span aria-hidden="true" class="glyphicon glyphicon-remove"></span>
							</button>
							<h4 class="modal-title">数据类别占比 (${taskId})</h4>
						</div>
						
						<div class="modal-body task-modal-body" style="height:calc(100% - 100px) !important;max-height:900px;overflow-x:auto;">
							<div>
								<div>
									<table id="exampleInfo1" class="table table-bordered table-hover">
							            <thead style="font-weight:bold;">
							                <tr>
												<th>id</th>
												<th>name</th>
												<th>train</th>
												<th>val</th>
												<th>test</th>
												<th>操作</th>
							                </tr>
							            </thead>
								        <tbody>
							                ${_data.data.map(m => `
									            <tr>
										            <td title="${m.id}">${m.id}</td>
										            <td title="${m.name}">${m.name}</td>
										            <td style="padding:0px;height:100%;line-height:250%;">
										            	${m.train.map(t => `
										            		<span>${t}</span>
										            	`).join('')}
										            </td>
										            <td style="padding:0px;height:100%;line-height:250%;">
										            	${m.val.map(t => `
										            		<span>${t}</span>
										            	`).join('')}
										            </td>
										            <td style="padding:0px;height:100%;line-height:250%;">
										            	${m.test.map(t => `
										            		<span>${t}</span>
										            	`).join('')}
										            </td>
										            <td style="padding:0px;height:100%;line-height:250%;">
										            	<button type="button" class="btn btn-success" onclick="myData.add_data('${taskId}','${m.id}')">增加</button>
										            </td>
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
//		table列表的属性样式等
    	$('#exampleInfo1').DataTable({
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

//	增加DATA
	add_data: function(task_id,class_id){
		var json_data = {
			"task_id": task_id,
			"class_id":class_id
		};
		$("#loading").css("display","block");
		var _url = trainUrl+"/add_data";
		util.newAjax(_url,"POST",json_data,function(data){
			$("#loading").css("display","none");
			if( data.errno == 1 ){
				util.errorView("成功"+data.message);
			}else{
				util.errorView("失败"+data.message);
			}
		});
	}

}
