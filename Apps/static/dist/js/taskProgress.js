
//点击任务进度
$(".taskProgress").click(function(){
	taskProgress.Table();
})
//任务进度管理
var taskProgress = {
	
//	任务进度
	Table: function(){

//  	任务进度列表的模板
	    var listModel = ` 
	    <div class="row">
		    <div class='testmodal'></div>
        	<div class="col-xs-12">
        		<div class="box box-primary">
        			<div class="box-header">
        				<h3 class="box-title">任务进度</h3>
    				</div>
	    			<div class="box-body">
						<table id="example2" class="table table-bordered table-hover">
					        <thead>
						        <tr>
									<th>task_id</th>
									<th>picid</th>
									<th>id_code</th>
									<th>statue</th>
									<th>start_time</th>
									<th>end_time</th>
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
	        "lengthMenu"  : [10],
	        "aaSorting"	  : false,
        	'processing'  : true,  								//隐藏加载提示,自行处理
        	'serverSide'  : false,  								//启用服务器端分页
	        ajax: function (data, callback, settings) {
	        	var param = {};
//				封装请求参数
	            param.limit = 5;								//页面显示记录条数，在页面显示每页显示多少项的时候
	            param.start = data.start;						//开始的记录序号
	            param.page = (data.start / data.length)+1;		//当前页码
				var _url = trainUrl+"/label_tasks";
//				使用post请求模型列表(每次只请求十条任务,然后对其进行分页)
	            $.ajax({
			        type : "get",
			        url : _url,
			        async : true,
//					contentType: "application/json; charset=utf-8",
			        data : {},
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
//      		第1列----task_id----显示
		        { 
	            	"data": "task_id",
					"class": "task_id"
	           	},
//      		第2列----picid----显示
		        { 
	            	"data": "picid",
					"class": "picid"
	           	},
//      		第3列----id_code----显示
		        { 
	            	"data": "id_code",
					"class": "id_code"
	           	},
//      		第4列----statue----显示
		        { 
	            	"data": "statue",
					"class": "statue"
	           	},
//      		第5列----start_time----显示
		        { 
	            	"data": "start_time",
					"class": "start_time"
	           	},
//      		第6列----end_time----显示
		        { 
	            	"data": "end_time",
					"class": "end_time"
	           	},
//      		第13列----操作按钮----显示(预览/删除/清除)
	            { 
	            	"data": "task_id",
	              	"render": function ( data, type, row,  meta ) {
						var centerModel = `
							<div class="btn-group">
								<button type="button" class="btn btn-warning details" title="${data}" onclick="taskProgress.details('${data}')">查看</button>
		                    </div>
						`;
				    	return centerModel;
					},
					"class": "center"
	            }
	        ]
	    });
	},
	
//	查看详情
	details: function(id){
		var _url = trainUrl+"/label_status?task_id="+id;
		$("#loading").css("display","block");
        $.ajax({
	        type : "get",
	        url : _url,
	        async : true,
	        data : {},
            success: function (data) {
				$("#loading").css("display","none");
            	var _data = data.data,
            		p_data = [];
            	if(!_data){
            		util.errorView(data.message);
            		return;
            	}
            	for(var item in _data){
            		p_data.push({
            			name: item,
            			value: _data[item]
            		})
            	}
			    var templ =`
					<div class="modal fade task-modal" id="modal-default">
						<div class="modal-dialog ftp-dialog" style="width: 80%;">
							<div class="modal-content task-modal-content" style="width: 30%;overflow: hidden;font-size: 18px;">
								<div class="modal-header">
				    				<button type="button" class="close" title="关闭" onclick="main.modalClose()">
				    					<span aria-hidden="true">&times;</span>
				    				</button>
									<h4 class="modal-title">任务进度详情</h4>
								</div>
								<div class="modal-body task-modal-body">
			      					<div class="box-body">
						                ${p_data.map(p => `
		    								<p>${p.name}: ${p.value}</p>
						                `).join('')}
			          				</div>
								</div>
							</div>
						</div>
					</div>
			    `;
			    $('.testmodal').html(templ);
			    $('#modal-default').modal('show');
            }
       	});
	}
}
