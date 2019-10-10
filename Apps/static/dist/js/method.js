
//使用jq来封装组件
(function ($) {
//	列表dom界面-重复使用
	$.addModel = function(dom, dataForms){
		var tableUrl = dataForms.url;				//获取列表数据的url
		var buttons = dataForms.buttonHtml;			//是否存在button操作按钮
		var eventClick = dataForms.event;			//button操作按钮的执行事件
		var objChild = dataForms.objChild;			//最上层的对象--即所要执行的事件在哪个对象下
		var ajaxType = dataForms.type || "get";			//最上层的对象--即所要执行的事件在哪个对象下
		var parms = JSON.stringify(dataForms.parms) || {};			//最上层的对象--即所要执行的事件在哪个对象下
		var columns = [];							//建立 数据列表的 数组
		for(var i=0; i<dataForms.list.length; i++){
			var value = dataForms.list[i].id;
			var type = dataForms.list[i].type;
			if(type && (type=="time")){				//数据中需要时间展示时的格式处理
				columns.push(
		            { 
		            	"data": value || "",
		              	"render": function ( data, type, row, meta ) {
		              		var time = util.Time(data);
						    return	time || "";
						}, 
						"class": value
					}
		        )
			}else if(type && (type=="event")){		//数据中需要button操作按钮的dom节点处理
				columns.push(
		            { 
		            	"data": value || "",
		              	"render": function ( data, type, row, meta ) {
		              		var taskId = row.task_id;
							var centerModel = `
								<div class="btn-group">
					                ${buttons.map(t => `
										<button type="button" class="btn btn-warning" name="${t.name}" title="${taskId}" data-toggle="modal" data-target="#modal-default">${t.text}</button>
					                `).join('')}
			                    </div>
							`;
					    	return centerModel;
						},
						"class": "centerModel"
					}
		        )
			}else{									//数据中得普通参数展示
				columns.push(
		            { 
		            	"data": value || "",
						"class": value
					}
		        )
			}
		}
		
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
				      		创建训练任务
				      	</button>
				    </div>
	    			<div class="box-body">
						<table id="example2" class="table table-bordered table-hover">
					        <thead>
						        <tr>
					                ${dataForms.list.map(d => `
										<th>${d.name}</th>
					                `).join('')}
						        </tr>
					        </thead>
		    			</table>
	    			</div>
	    		</div>
	    	</div>
	    </div>
	    `;

	    $(dom).html(listModel);
		
//		初始化表格(训练任务列表的属性)
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
//				var url = trainUrl+"/train_history?page=1";
	            $.ajax({
			        type : ajaxType,
			        url : tableUrl,
			        async : true,
					contentType: "application/json; charset=utf-8",
			        data : parms,
	                success: function (data) {
	                	taskData1 = data.data;
	                    var returnData = {};
	                    returnData.recordsTotal = data.data.length;				//返回数据全部记录
	                    returnData.recordsFiltered = data.data.length;			//后台不实现过滤功能，每次查询均视作全部结果
	                    returnData.data = data.data;						//返回的数据列表
                    	callback(returnData);
                    	eventClick(objChild);
	                }
	           	});
	        },
	        
//			我的任务列表的数据展示
        	columns: columns
	    });
	};
	
//	使用ajax的方式get获取请求数据
	$.getAjax = function(url,callback){
    	$.ajax( {
        	type : "GET",
	        url : url,
	        async : true,
	        data : {},
	        success : function(msg) {
	        	if(callback){
	        		callback(msg);
	        	}
	        },
		   	error: function(XMLHttpRequest, textStatus, errorThrown) {
		   		console.log(XMLHttpRequest.status+"---"+XMLHttpRequest.readyState+"---"+textStatus)
		   	},
	    });
	};
	
//	使用ajax的方式post获取或提交数据
	$.postAjax = function(url,form,callback){
	    $.ajax( {
	        type : "POST",
	        url : url,
        	async : true,
			contentType: "application/json; charset=utf-8",
	        data : JSON.stringify(form),
	        success : function(msg) {
	        	callback(msg);
	        },
	        error: function(msg){
	            util.errorView("系统异常");
	        }
	    });
	};
	
//	时间格式方法(包括时分秒)
	$.Time = function(value,type=0){
		var dateTime = new Date(value);
	    var y = dateTime.getFullYear();			//年
	    var m = dateTime.getMonth() + 1;		//月
	    m = m < 10 ? '0' + m : m;
	    var d = dateTime.getDate();				//日
	    d = d < 10 ? ('0' + d) : d;
	    var h = dateTime.getHours();			//时
	    h = h < 10 ? ('0' + h) : h;
	    var t = dateTime.getMinutes();			//分
	    t = t < 10 ? ('0' + t) : t;
	    var s = dateTime.getSeconds();			//秒
	    s = s < 10 ? ('0' + s) : s;
	    var time = y + '-' + m + '-' + d;
	    if(type == 1){							//type类型为1的时候为      年月日时分秒
	    	time =  y + '-' + m + '-' + d + ' ' + h + ':' + t + ':' + s;
	    }else if(type == 2){					//type类型为2的时候为      年月日时分秒 毫秒
	    	time =  y + '-' + m + '-' + d + ' ' + h + ':' + t + ':' + s + '.' + dateTime.getMilliseconds();
	    }else{									//type类型为其他的时候为      年月日
	    	time = y + '-' + m + '-' + d;
	    }
		return time;
	};
	
//	新增一个polygon的面------任务号，坐标点，颜色，总里程数，有效里程数，透明度
	$.newPolygon = function(taskId,locs,fill,totalMileage=false,effectiveMileage=false,opacity=0.2){
		var nodes = [];
		var firstPoint = new Point({ loc : locs[0] });
		nodes.push( firstPoint );
		for(var i=1; i<locs.length-1; i++){
			nodes.push( new Point({ loc : locs[i] }) );
		}
		nodes.push( firstPoint );
		var polygon = new Polygon({
			id : ('polygon_'+taskId),
			nodes : nodes,
			mode: 'polygon',
			enable : true,
            style:({
           		"fill": fill,
           		'opacity':opacity,
           		"stroke": 'black',
           		"stroke-width" : 3
           	}),
           	"tag": {
	           	"taskId" : taskId,
	           	"totalMileage" : totalMileage || "",
	           	"effectiveMileage" : effectiveMileage || ""
           	}
		});
        return polygon;
	};
	
//  polygon - 范围的坐标点格式修改
    $.rangePolygon = function(range){
		if( !range || (range.indexOf("POLYGON") < 0) ){
			return [];
		}
		var replaces = range.replace("POLYGON((","").replace("))","");
		var Coordinate = replaces.split(",");
//		添加任务范围显示
		var locs = [];
		for(var i=0; i<Coordinate.length; i++){
			var strs2 = Coordinate[i].split(" ");
			var loc = []; 								//定义一数组 
			for(var l=0; l<strs2.length; l++){
				loc.push( Number(strs2[l]) );
			}
			locs.push(loc);
		}
	    return locs;
    };
    
//	新增一个polyline的线-轨迹------任务号，轨迹号，颜色，坐标点，轨迹名，箭头指向(道路方向)，总里程，有效里程，透明度，完成状态，识别状态
	$.newPolyline = function(taskId,trackId,styleColor,locs,trackName="",arrow=false,totalMileage=false,effectiveMileage=false,judge=false,color=false){
		var finished = false;
		if(judge && createTasks.workerTasks[trackId] && !color){
			finished = true;
			styleColor = createTasks.updateStyleFinish;
		}
		var nodes = [];
		for(var i=0; i<locs.length; i++){
			nodes.push( new Point({ loc : locs[i] }) );
		}
        var polyline = new Polyline({
        	id : ('polyline_'+trackId+"_"+taskId),
        	nodes : nodes,
        	mode : 'polyline',
			enable : false,
			arrow : false,							//箭头方向
            onDraw : styleColor,
           	"taskId" : taskId,
            'trackId':trackId,
           	"trackName" : trackName,
           	"totalMileage" : totalMileage,
           	"effectiveMileage" : effectiveMileage,
           	"finished" : finished,
           	"color" : color
        });
        return polyline;
	};
    
//  polyline - 轨迹的坐标点格式修改
	$.rangePolyline = function(polyline){
		if(!polyline){
			return [];
		}
		var replaces = polyline.replace("LINESTRING(","").replace(")","");
		var Coordinate = replaces.split(",");
//		添加任务范围显示
		var locs = [];
		for(var i=0; i<Coordinate.length; i++){
			var strs2 = Coordinate[i].split(" ");
			var loc = []; 								//定义一数组 
			for(var l=0; l<strs2.length; l++){
				loc.push( Number(strs2[l]) );
			}
			locs.push(loc);
		}
	    return locs;
	};
	
//	更新地图覆盖物的样式
	$.uploadLayers = function(colorArr=false){
		var map = window.editor.context.map();
		map.redraw();
	};
	
//	打开任务监控界面--流程界面-------流程的实例id-任务流程id（查询任务的详情）
	$.flowingModale = function(processInstanceId,activityId,url,type){
//  	预览任务(打开监控界面),需要获取该任务的详情数据
		var processDefinitionId,taskId,taskName;
		$.getAjax(url,function(data){
//			获取所需要的监控界面所需要的参数
			processDefinitionId = data.data[0].processDefinitionId;
			for(var l=0; l<data.data[0].variables.length; l++){
				if(data.data[0].variables[l].name == "taskId"){
					taskId = data.data[0].variables[l].value;					//任务的taskId
				}
				if(data.data[0].variables[l].name == "taskName"){
					taskName = data.data[0].variables[l].value;			//任务名
				}
			}
//			将所需参数存入数组当中
			monitor.monitorIds = {
				taskName			: "当前任务 ( "+taskId+"--"+taskName+" )",
				processDefinitionId : processDefinitionId,
				processInstanceId 	: processInstanceId,
				taskId 				: taskId
			}
			monitor.monitorInit(type);
		})
	};
	
//	刷新列表的功能
	$.refreshTable = function(colorArr=false){
		$("#refresh button").attr("disabled", true);
		$("#refresh button").html("刷新列表...");
		setTimeout(function(){
			value.Table();
		},100)
	};
	
//	提示信息（错误信息及成功提示）
	$.information = function(view,junge=false){
		var className = "errorUpload";
		if(junge){
			className = "successpload";
		}
		var errorModel = `
			<span class="${className}" title="提示信息:${view}!">提示信息:${view}!</span>
		`;
		$(".upload").html(errorModel);
		$(".upload").fadeIn(300);
		setTimeout(function(){
			$(".upload").fadeOut(300);
		},1500);
	};
	
//	组件封装
	/*
	 * 
	$.information = function(view,junge=false){
		var className = "errorUpload";
		if(junge){
			className = "successpload";
		}
		var errorModel = `
			<span class="${className}" title="提示信息:${view}!">提示信息:${view}!</span>
		`;
		$(".upload").html(errorModel);
		$(".upload").fadeIn(300);
		setTimeout(function(){
			$(".upload").fadeOut(300);
		},1500);
	};
	*
	*/
	
	
})(jQuery);



