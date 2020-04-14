
//点击Network
$(".myNetwork").click(function(){
	myNetwork.Table();
})
//Network管理
var myNetwork = {
//	Network的模版列表
	Table: function(){
		
//  	Network的模板
	    var listModel = ` 
	    <div class="row">
		    <div class='testmodal'></div>
        	<div class="col-xs-12">
        		<div class="box box-primary">
        			<div class="box-header">
        				<div class="buttons"></div>
        				<h3 class="box-title">Network</h3>
    				</div>
    				<!--
				    <div class="taskButtons" style="padding-left:10px;">
				      	<button type="button" onclick="myNetwork.createModel();" class="btn btn-success">
				      		创建network
				      	</button>
				    </div>
				    -->
	    			<div class="box-body">
						<table id="example2" class="table table-bordered table-hover">
					        <thead>
						        <tr>
					                <th>ID</th>
					                <th>Network</th>
					                <th>描述</th>
						        </tr>
					        </thead>
		    			</table>
	    			</div>
	    		</div>
	    	</div>
	    </div>
	    `;

	    $('.content.container-fluid').html(listModel);

//		初始化表格(Network的属性)
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
			
//				使用post请求Network(每次只请求十条任务,然后对其进行分页)
	            $.ajax({
			        type : "get",
			        url : trainUrl+"/network_history",
			        async : true,
			        data : {},
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
//			Network的数据展示
        	columns: [
//      		第一列----id----显示
	            { 
	            	"data": "id" || "",
					"class": "id"
				},
//      		第二列----network----显示
	            { 
	            	"data": "net_name" || "",
					"class": "net_name"
				},
//      		第三列----描述----显示
	            { 
	            	"data": "net_describe" || "",
					"class": "net_describe"
				}
	        ]
	            
	    });
	},
	

//	创建netWork
	createModel: function(){
		var newData = [
			{
				"name":"network名称",
				"id":"network_name"
			},
			{
				"name":"源文件",
				"id":"src_net",
				"type":"select",
				"url":trainUrl+"/src_network"
			},
			{
				"name":"network描述",
				"id":"net_descr",
				"type":"text"
			}
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
							<h4 class="modal-title">创建netWork</h4>
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
							                <button type="button" class="btn btn-info pull-right" title="创建Network">创建Network</button>
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
	    	myNetwork.submitForm(newData)
	    })
	},
	
	format: function(data){
		for(var i=0; i<data.length; i++){
			var title = data[i].text || data[i].name;
			var dataValue = data[i].value || "";
			if(data[i].title){
				title = data[i].name + data[i].title;
			}
			if(data[i].type == "select"){
				var formArr = [];
				if(data[i].url && !data[i].refresh){
					var getData = util.newAjax(data[i].url,"GET") || {"data":[]};
					for(var l=0; l<getData.data.length; l++){
						formArr.push({
							"name": getData.data[l],
							"value": getData.data[l]
						})
					}
				}else if(!data[i].url){
					formArr = data[i].data;
				}
				var refreshButton = ``;
				var refreshButtonNum = 6;
				if(data[i].refresh && data[i].url){
					refreshButton = `
						<div class="col-sm-1" style="padding:0;height:34px;line-height:34px;">
							<a id="rightBtn" class="glyphicon glyphicon-chevron-right" title="选择任务" href="#" name="${data[i].id}" style="color:gray" onclick="startTask.refresh('${data[i].id}','${data[i].url}')"></a>
						</div>
					`;
				}
				var options = "";
				if(!data[i].typeLable){
					options = `<option class="change" value="">${data[i].name}</option>`;
				}
				data[i].html = `
					<label for="${data[i].id}" class="col-sm-5 control-label" style="text-align:left;">${data[i].name} :</label>
					<div class="col-sm-7">
						<select class="form-control" id="${data[i].id}" title="${data[i].name}">
				            ${options}
			                ${formArr.map(d => `
				            	<option class="change" value="${d.value}">${d.name}</option>
			                `).join('')}
			            </select>
			            ${refreshButton}
					</div>
				`;
			}else if(data[i].type == "text"){
				var textValue = "";
				if(data[i].url){
					var getData = util.newAjax(data[i].url,"GET") || {"data":[]};
					textValue =  util.formatJson(getData.data);
				}
				data[i].html = `
					<label for="${data[i].id}" class="col-sm-5 control-label" style="text-align:left;">${data[i].name} :</label>
					<div class="col-sm-7">
	    				<textarea id="${data[i].id}" style="height:150px;width:200px;" title="${data[i].name}">${textValue}</textarea>
					</div>
				`;
			}else if(data[i].type == "checkbox"){
				data[i].html = `
					<label title="${data[i].name}" style="padding:5px 15px;">
						<input type="checkbox" id="${data[i].id}" value="" onclick="createTask.Continue('${data[i].event || ""}','${data[i].id}')" />
						<span style="float:left;"></span>
						<b style="padding-left:5px;font-weight:normal;" title="${data[i].name}">${data[i].name}</b>
					</label>
				`;
			}else{
				data[i].html = `
					<label for="${data[i].id}" class="col-sm-5 control-label" style="text-align:left;">${data[i].name} :</label>
					<div class="col-sm-7">
						<input type="text" title="${title}" class="form-control" id="${data[i].id}" value="${dataValue}" placeholder="${title}" onfocus="this.placeholder=''" onblur="this.placeholder='${title}'">
					</div>
				`;
			}
		}
		return data;
	},
	
//	提交创建netWork
	submitForm: function(newData){
		var jsonObj = {};
		for(var i=0; i<newData.length; i++){
			var id = newData[i].id;
			var value = $("#"+id).val() || $("#"+id+" option:selected").val() || "";
			jsonObj[id] = value;
		}
		var url = trainUrl+"/create_net";
		util.newAjax(url,"POST",jsonObj,function(reuslt){
			if( result.errno == 1 ){
				util.errorView(result.message,true);
				main.modalClose();
			}else{
				console.log(reuslt);
				util.errorView(result.message);
		        return;
			}
		});
	}
}
