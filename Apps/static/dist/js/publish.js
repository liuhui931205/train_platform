
//点击发布
$(".publish").click(function(){
	publish.Table();
})
//发布管理
var publish = {
	
	listForm: [
		{
			"name":"任务名",
			"text":"任务名",
			"id":"output_dir"
		},
		{
			"name":"筛选张数",
			"text":"筛选张数",
			"id":"ratio"
		},
		{
			"name":"gpu",
			"text":"gpu,以逗号分隔",
			"id":"gpus"
		},
		{
			"name":"模型地址",
			"id":"weights_dir",
			"type":"select",
			"url":trainUrl+"/model_lists"
		},
		{
			"name":"track_file地址",
			"text":"track_file地址",
			"id":"track_file"
		},
		{
			"name":"taskId",
			"text":"taskId,以逗号分隔",
			"id":"task_file"
		}
	],
	
	modelType: [
		{
			"name" : "车道线",
			"value": "1"
		},
		{
			"name" : "路牌",
			"value": "2"
		},
		{
			"name" : "灯杆",
			"value": "3"
		},
		{
			"name" : "虚拟车道线",
			"value": "4"
		},
		{
			"name" : "箭头识别",
			"value": "5"
		},
		{
			"name" : "光流",
			"value": "6"
		},
		{
			"name" : "旧路牌",
			"value": "7"
		}
	],
	
//	所获取到的所有已发布的模型缓存
	modelJson:{
		
	},
	
//	发布的模版列表
	Table: function(modelValue="1",modelName="车道线"){
		publish.modelJson = {};
//  	发布列表的模板
	    var listModel = ` 
	    <div class="row">
		    <div class='testmodal'></div>
        	<div class="col-xs-12">
        		<div class="box box-primary">
        			<div class="box-header">
        				<div class="buttons"></div>
        				<h3 class="box-title">发布</h3>
    				</div>
    				
					<div class="modelType" style="padding-left:10px;float:left;margin-right:10px;">
						<select class="form-control" style="width:200px;float:left;">
			                ${publish.modelType.map(m => `
				            	<option value="${m.value}">${m.name}</option>
			                `).join('')}
			            </select>
			        </div>
    				
				    <div class="taskButtons" style="padding-left:10px;">
				      	<button type="button" class="btn btn-success evaluate">评估</button>
				      	<button type="button" class="btn btn-success releaseModel">发布</button>
				      	<button type="button" class="btn btn-success autoImage" >自动挑图</button>
				    </div>
	    			<div class="box-body">
						<table id="example2" class="table table-bordered table-hover">
					        <thead>
						        <tr>
					                <th>${modelName}模型<span class="fine"> (默认选中当前展开的模型)<span></th>
						        </tr>
					        </thead>
		    			</table>
	    			</div>
	    		</div>
	    	</div>
	    </div>
	    `;

	    $('.content.container-fluid').html(listModel);

	    var modelSelect = $('.modelType select option');
//		记忆上次搜索操作的保存
		for(var i=0; i<modelSelect.length; i++) {
			if(modelSelect[i].value == modelValue) {
				modelSelect[i].selected = true;
			}
		}
//		初始化表格(发布列表的属性)
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
				var url = trainUrl+"/release_tab?page="+modelValue;
			
//				使用post请求模型列表(每次只请求十条任务,然后对其进行分页)
	            $.ajax({
			        type : "get",
			        url : url,
			        async : true,
			        data : {},
	                success: function (data) {
	                	taskData1 = [];
	                    var returnData = {};
	                    for(var item in data.data){
	                    	taskData1.push({
	                    		"text": item,
	                    		"nodes": data.data[item]
	                    	})
	                    }
	                    returnData.recordsTotal = taskData1.length;				//返回数据全部记录
	                    returnData.recordsFiltered = taskData1.length;			//后台不实现过滤功能，每次查询均视作全部结果
	                    returnData.data = taskData1;						//返回的数据列表
//						调用DataTables提供的callback方法，代表数据已封装完成并传回DataTables进行渲染
//						此时的数据需确保正确无误，异常判断应在执行此回调前自行处理完毕
                    	callback(returnData);
	                }
	           	});
	        },
	        
//			发布列表的数据展示
        	columns: [
//      		第一列----模型名称----显示
		        { 
	            	"data": "text",
	              	"render": function ( data, type, row,  meta ) {
	              		var nameModel = `
	              			<span title="${data}">${data}</span>
	              		`;
	              		
	              		return nameModel;
					},
					"class": "modelChild"
	           	}
	        ]
	            
	    });
	    
//		单个服务可能会存在多个实例--展开实例
	    $('#example2 tbody').on('click', 'tr td.modelChild', function () {
	        var tr = $(this).closest('tr');
	        var row = table.row( tr );
	        if ( row.child.isShown() ) {
	            row.child.hide();
	            tr.removeClass('shown');
	        }else{
	            if(!row.data()){
	            	return;
	            }
    			var textName = row.data().text,
    				nodes = row.data().nodes;
    			delete publish.modelJson[textName];						//重新加载模型
	            row.child( myModel.format(textName,nodes) ).show();
	            tr.addClass('shown');
	            tr[0].nextSibling.style.backgroundColor = "#F0F0F0";
	        }
	    });
	    
//		点击自动挑图
	    $('.autoImage').click(function(){
	    	publish.autoImage()
	    });
	    
//		点击发布触发
	    $('.releaseModel').click(function(){
			var modelValue = $(".modelType option:selected").val();
	        var params = {
	            "dicts": publish.modelJson,
	            "page": modelValue
	        }
	    	publish.releaseModel(params)
	    });
	    
//		点击模型类型触发
	    $('.modelType select').change(function(){
	        publish.modelChange();
	    });

//		去执行评估事件
	    $('.evaluate').click(function(){
	    	myModel.evaluate(true);
	    });
	    
//		点击checkbox选中某种模型
	    $('#example2 tbody').on('click', '.checkboxs input', function () {
	    	var thisName = this.name;
			publish.modelJson[thisName] = [];
			var doms = $("."+thisName+" .checkboxs input");
			for(var i=0; i<doms.length; i++){
				var checked = doms[i].checked;
				var value = doms[i].value;
				if(checked){
					publish.modelJson[thisName].push(value)
				}
			}
	    });
	},
	
//	模型详情点击后的html
	format: function(textName,nodes){
	    if(nodes.length > 0){
			var model = `
				<h4 class="infoHeader">模型信息</h4>
				<table id="instanceChild" class="table table-bordered ${textName}" style="margin-bottom:0px;">
			        <tbody style="background:#fff;">
		                ${nodes.map(d => `
				            <tr>
				            	<td class="checkboxs" title="${d}">
									<label title="${d}">
										<input type="checkbox" name="${textName}" value="${d}" />
										<span></span>
									</label>
				            	</td>
				            	<td title="${d}">
				            		${d}
				            	</td>
				            </tr>
		                `).join('')}
			        </tbody>
			    </table>
			`;
	    }else{
			var model = `
		    	空
			`;
	    }
	    return model;
	},
	
//	选择模型类型的select选择变化时,对列表数据进行修改
	modelChange: function(type=false){
		var modelValue = $(".modelType option:selected").val();
		var modelName = $(".modelType option:selected").html();
		if(type){
			myModel.Table(modelValue,modelName);
		}else{
			publish.Table(modelValue,modelName);
		}
	},
	
//	通过已发布的模型进行自动挑图
	autoImage: function(){
		var dataForm = myNetwork.format(publish.listForm);
		var createModel =`
		<div class="modal fade task-modal" data-backdrop="static" data-show="true" id="modal-default">
			<div class="modal-dialog ftp-dialog">
				<div class="modal-content task-modal-content">
					<div class="modal-header">
	    				<button type="button" class="close" title="关闭" data-dismiss="modal" aria-label="Close" onclick="main.modalClose()">
	    					<span aria-hidden="true">&times;</span>
	    				</button>
						<h4 class="modal-title">自动挑图</h4>
					</div>
					<div class="modal-body task-modal-body">
					    <div class="right-content" style="margin:0 auto;width:300px;">
							<form id='taskForm2' class="form-horizontal">
		      					<div class="box-body">
					                ${dataForm.map(f => `
										<div class="form-group">
											${f.html}
										</div>
					                `).join('')}
									<div class="form-group">
										<label title="是否打乱顺序" style="padding:5px 15px;">
											<input type="checkbox" id="isshuffle" value="true" onclick="console.log('是否打乱顺序')" />
											<span style="float:left;"></span>
											<b style="padding-left:5px;font-weight:normal;" title="是否打乱顺序">是否打乱顺序</b>
										</label>
									</div>
		          				</div>
		      					<!-- 确定与取消按钮 -->
		      					<div class="box-footer">
					                <button type="button" class="btn btn-default" title="取消" onclick="main.modalClose()">取消</button>
					                <button type="button" class="btn btn-info pull-right" title="提交" onclick="publish.submitImage()">提交</button>
		      					</div>
		    				</form>
		  				</div>
					</div>
				</div>
			</div>
		</div>
		`;
		$('.testmodal').html(createModel);
	    $('#modal-default').modal('show');
	},
	
//	开始自动挑图
	submitImage: function(){
		var parmsData = {};
		for(var i=0; i<publish.listForm.length; i++){
			var id = publish.listForm[i].id;
			var value = $("#"+id).val();
			parmsData[id] = value;
		}
		parmsData["isshuffle"] = Number( $("#isshuffle")[0].checked );
		
        var submitUrl = trainUrl+"/sele_value";
        var processUrl = trainUrl+"/auto_sele";
		util.newAjax(submitUrl,"POST",parmsData,function(result){
//			成功时进行进度条转换( resp.errno=='1'的时候 )
//			否则为发布失败   -- 进行提示
			if( result.errno == 1 ){
				util.errorView(result.message,true);
				main.modalClose();
				createTask.taskProgress(result.data.task_id,processUrl);
			}else{
				console.log(result);
				util.errorView(result.message);
		        return;
			}
		});
	},
	
//	触发发布事件--对选中的模型进行发布
	releaseModel: function(params){
		myModel.nextRelease(params);
//      var url = trainUrl+"/sessions";
//		util.newAjax(url,"POST",params,function(reuslt){
//			if( result.errno == "1" ){
//				myModel.nextRelease();
//			}else{
//				console.log(reuslt);
//				util.errorView("系统异常");
//		        return;
//			}
////			成功时转到输入form表单--发布成功时( resp.errno=='1'的时候 )
////			否则为发布失败   -- 进行提示
////          alert(resp.errmsg)
//		});
	}
}
