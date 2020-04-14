
//点击模型
$(".myModel").click(function(){
	myModel.Table();
})
//模型管理
var myModel = {

	modelPage: "1",

	taskId: null,

	modelType: [
		{
			"name" : "语义分割",
			"value": "1"
		},
		{
			"name" : "分类模型",
			"value": "2"
		},
		{
			"name" : "目标检测",
			"value": "3"
		},
		{
			"name" : "OCR",
			"value": "4"
		}
	],

//	模型的模版列表
	Table: function(modelValue="1",modelName="语义分割"){
		myModel.modelPage = modelValue;
		publish.modelJson = {};
//  	模型列表的模板
	    var listModel = ` 
	    <div class="row">
        	<div class="testmodal"></div>
        	<div class="col-xs-12">
        		<div class="box box-primary">
        			<div class="box-header">
        				<div class="buttons"></div>
        				<h3 class="box-title">模型管理</h3>
    				</div>
    				
					<div class="modelType" style="padding-left:10px;float:left;margin-right:10px;">
						<select class="form-control" style="width:200px;float:left;">
			                ${myModel.modelType.map(m => `
				            	<option value="${m.value}">${m.name}</option>
			                `).join('')}
			            </select>
			        </div>
				    <div class="taskButtons" style="padding-left:10px;">
				      	<button type="button" class="btn btn-success evaluate">
				      		评估
				      	</button>
				      	<button type="button" class="btn btn-success release">
				      		发布
				      	</button>
				      	<button type="button" class="btn btn-success _level">
				      		置信度
				      	</button>
				    </div>
	    			<div class="box-body">
						<table id="example2" class="table table-bordered table-hover">
					        <thead>
						        <tr>
					                <th>${modelName}模型名称<span class="fine"> (默认选中当前展开的模型)<span></th>
						        </tr>
					        </thead>
		    			</table>
	    			</div>
	    		</div>
	    	</div>
	    </div>
	    `;

	    $('.content.container-fluid').html(listModel);

//		记忆上次搜索操作的保存
	    var modelSelect = $('.modelType select option');
		for(var i=0; i<modelSelect.length; i++) {
			if(modelSelect[i].value == modelValue) {
				modelSelect[i].selected = true;
			}
		}

//		初始化表格(模型列表的属性)
    	var table = $("#example2").DataTable({
	    	'language'    : main.lang,
	        'searching'   : false,								//原生搜索
	        'paging'      : true,
	        'lengthChange': false,
	        'ordering'    : false,
	        'info'        : true,
	        'autoWidth'   : false,
	        'scrollY'		: 700,
	        'scrollCollapse': true,
	        "aaSorting"	  : false,
        	'processing'  : true,  								//隐藏加载提示,自行处理
        	'serverSide'  : false,  								//启用服务器端分页
	        ajax: function (data, callback, settings) {
	        	var param = {};
//				封装请求参数
	            param.limit = 10;								//页面显示记录条数，在页面显示每页显示多少项的时候
	            param.start = data.start;						//开始的记录序号
	            param.page = (data.start / data.length)+1;		//当前页码
				var url = trainUrl+"/model?page="+modelValue;

//				使用post请求模型列表(每次只请求十条任务,然后对其进行分页)
	            $.ajax({
			        type : "get",
			        url : url,
			        async : true,
			        data : {},
	                success: function (data) {
	                	var taskData1 = [];
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

//			模型列表的数据展示
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
	           	},
	        ]
	    });

//		点击模型类型触发
	    $('.modelType select').change(function(){
	        publish.modelChange(true);
	    });

//		单个服务可能会存在多个实例--展开实例
	    $('#example2 tbody').on('click', 'tr td.modelChild', function () {
	        var tr = $(this).closest('tr');
	        var row = table.row( tr );
	        if ( row.child.isShown() ) {
	        	var textName = row.data().text;
    			delete publish.modelJson[textName];
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
//		去执行评估事件
	    $('.evaluate').click(function(){
	    	myModel.evaluate();
	    })
//		去执行发布事件
	    $('.release').click(function(){
	        var params = {
	            "dicts": publish.modelJson,
	            "page": ""
	        }
	    	publish.releaseModel(params)
	    });
//		去执行发布事件
	    $('._level').click(function(){
			var arr = [],
				params = {};
			for(id in publish.modelJson){
				arr.push(id);
				params = {
					"task_name": id,
					"model": publish.modelJson[id][0]
				}
			}
			if(arr.length > 1){
				util.errorView("不支持多个模型查询");
				return;
			}
	    	myModel._levelModel(params)
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

//	置信度界面功能开发
	_levelModel: function(params){
		console.log(params);
		util.newAjax(trainUrl+"/con_info", "POST", params, function(data){
			if(!data.total_data){
				util.errorView(data.message);
				return;
			}
			var total_data = [],
				total_cls_con = [];
			for(var item in data.total_data){
				if(item == "total_cls_con"){
					for(var id in data.total_data[item]){
						total_cls_con.push({
							"id": id,
							"value": data.total_data[item][id]
						})
					}
				}else{
					total_data.push({
						"id": item,
						"value": data.total_data[item]
					})
				}
			}

			console.log(data)
		    var templ =`
				<div class="modal fade task-modal" id="modal-default">
					<div class="modal-dialog ftp-dialog" style="width: 80%;">
						<div class="modal-content task-modal-content" style="width: 100%;overflow: hidden;">
							<div class="modal-header">
			    				<button type="button" class="close" title="关闭" onclick="main.modalClose()">
			    					<span aria-hidden="true">&times;</span>
			    				</button>
								<h4 class="modal-title">模型置信度信息</h4>
							</div>
							<div class="modal-body task-modal-body">
		      					<div class="box-body">
							        <p style="margin-bottom: 0px;">
							        	<b>整体分类的置信度: </b>
						                ${total_cls_con.map(c => `
								            <span>${c.id}- ${c.value};   </span>
						                `).join('')}
							        </p>
					                ${total_data.map(d => `
							            <p style="margin-bottom: 0px;"><b>${d.id}:</b> ${d.value}</p>
					                `).join('')}
		      					
									<table id="exampleInfo" class="table table-bordered table-hover" style="margin-top:10px;white-space: nowrap;">
							            <thead style="font-weight:bold;">
							                <tr>
												<th>分类置信度</th>
												<th>整体置信度</th>
												<th>TrackPointId</th>
												<th>整体置信度（去除路面和其他）</th>
							                </tr>
							            </thead>
								        <tbody
							                ${data.con_data.map(m => `
									            <tr>
										            <td title="${m.origin_cls_con}">${m.origin_cls_con}</td>
										            <td title="${m.origin_whole_con}">${m.origin_whole_con}</td>
										            <td title="${m.trackpointid}">${m.trackpointid}</td>
										            <td title="${m.whole_con}">${m.whole_con}</td>
									            </tr>
							                `).join('')}
								        </tbody>
									</table>
		          				</div>
							</div>
						</div>
					</div>
				</div>
		    `;
		    $('.testmodal').html(templ);
		    $('#modal-default').modal('show');
		    setTimeout(function(){
		        $('#exampleInfo').DataTable({
					'language'    : window.lang,
		            'paging'      : false,
		            'lengthChange': false,
		            'searching'   : false,
		            'ordering'    : false,
			        'scrollY'		: 700,
			        'scrollCollapse': false,
		            'info'        : true,
		            'autoWidth'   : false
		        });
		    },500)
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

//	评估
	evaluate: function(judge=false){
        var params = {
            "dicts": publish.modelJson
        }
		myModel.nextEvaluate(params,judge);
	},

//	开始评估的form表单结构填写
	nextEvaluate: function(params,judge=false){
		var newData = {
			"1":[
				{
					"name":"挑选数据",
					"id":"type",
					"type":"select",
					"typeLable":true,
					"url":trainUrl+"/eva_data"
				},
				{
					"name":"图片类型",
					"id":"img",
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
					"name":"GPU",
					"id":"gpus",
					"title":" (用逗号分割)"
				},
				{
					"name":"single_gpu",
					"title":"例如：4",
					"id":"single_gpu"
				},
				{
					"name":"server",
					"title":"例如：10",
					"id":"host"
				},
				{
					"name":"置信度",
					"id":"output_confidence",
					"type":"checkbox",
				}
			],
			"2":[
				{
					"name":"分类模型",
					"id":"type",
					"type":"select",
					"data":[
						{
							"name":"val",
							"value":"val"
						},
						{
							"name":"test",
							"value":"test"
						},
						{
							"name":"train",
							"value":"train"
						}
					]
				},
				{
					"name":"图片类型",
					"id":"img",
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
					"name":"GPU",
					"id":"gpus",
					"title":" (用逗号分割)"
				},
				{
					"name":"置信度",
					"id":"output_confidence",
					"type":"checkbox",
				}
			],
			"3":[
				{
					"name":"挑选数据",
					"id":"type",
					"type":"select",
					"typeLable":true,
					"url":trainUrl+"/eva_data"
				},
				{
					"name":"图片类型",
					"id":"img",
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
					"name":"GPU",
					"id":"gpus",
					"title":" (用逗号分割)"
				},
				{
					"name":"single_gpu",
					"title":"例如：4",
					"id":"single_gpu"
				}
			]
		}
//		判断进入方式--模版评估  ,发布评估
		if( !judge && (myModel.modelPage=="1") ){
			newData[myModel.modelPage].push({
				"name":"测试集数据",
				"id":"l_value",
				"type":"checkbox"
			});
		}

		var data = myNetwork.format(newData[myModel.modelPage]);
		myModel.addModel(data,"评估")
	    $(".pull-right").click(function(){
	    	myModel.startEvaluate(params,newData[myModel.modelPage],"/starteva",true)
	    })
	},

//	开始发布的form表单结构填写
	nextRelease: function(params = {}){
		var newData = [
			{
				"name":"模型版本号",
				"text":"模型版本号",
				"id":"version"
			},
			{
				"name":"模型环境",
				"text":"模型环境,全图填写fullSize",
				"id":"env"
			},
			{
				"name":"adcode",
				"text":"adcode",
				"id":"model_code"
			},
			{
				"name":"类型",
				"id":"type",
				"type":"select",
				"data":[
					{
						"name":"车道线",
						"value":"resnet-road"
					},
					{
						"name":"路牌",
						"value":"sign-mask-rcnn"
					},
					{
						"name":"旧路牌",
						"value":"pspnet"
					},
					{
						"name":"灯杆",
						"value":"resnet1"
					},
					{
						"name":"虚拟车道线",
						"value":"virtual-lane"
					},
					{
						"name":"箭头识别",
						"value":"arrow"
					},
					{
						"name":"光流",
						"value":"flow-net"
					}
				]
			},
			{
				"name":"模型描述",
				"id":"desc",
				"type":"text"
			},
			{
			"name":"prod",
			"id":"prod",
			"type":"checkbox",
			}
		]
		var data = myNetwork.format(newData);
		myModel.addModel(data,"发布")
	    $(".pull-right").click(function(){
	    	myModel.startEvaluate(params,newData,"/release")
	    })
	},

//	开始评估--开始发布
	startEvaluate: function(params,newData,urlText,judge=false){
		for(var i=0; i<newData.length; i++){
			var id = newData[i].id;
			var value = $("#"+id).val() || $("#"+id+" option:selected").val() || "";
			if( newData[i].type && (newData[i].type == "checkbox") ){
				value = $("#"+id)[0].checked;
			}
			params[id] = value;
		}
		/*if(!judge){
			var dataTime = util.Time( (new Date()) );
			params["time"] = dataTime;
		}else{
			params["l_value"] = downLoad.l_value || '';
		}*/
        var url = trainUrl+urlText;
        var progressUrl = trainUrl+"/eva_rate";

		util.newAjax(url,"POST",params,function(result){
//			成功时关闭模态框
			if( result.errno == "1" ){
				myModel.taskId = result.data.task_id;
				util.errorView(result.message,true);
				main.modalClose();
//				判断是否为评估,成功则跳转到评估界面
				if(judge){
					downLoad.l_value = "";
					assessment.dir = params["dir"];
					assessment.dest_dir = params["dest_dir"];
					if(myModel.modelPage == "3"){
						myModel.jumpImage();
						return;
					}
					createTask.taskProgress(result.data.task_id,progressUrl,function(){
						myModel.show_info();
					});
				}
			}else{
				console.log(result);
				util.errorView("系统异常");
		        return;
			}
		});
	},

//	信息展示的按钮点击跳转评估界面
	jumpImage: function(){
		main.modalClose();
		$('.t-info a').click();
	},

//	模型评估完成后,查询并展示任务相关信息
	show_info: function(){
		var parmsData = {
			"task_id": myModel.taskId
		};
		var infoUrl = trainUrl+"/show_info",
			buttomModel = "";
		if(myModel.modelPage == "1"){
			buttomModel = `<button id="showInfo" class="btn btn-success" onclick="myModel.jumpImage();">图像评估</button>`;
		}
		util.newAjax(infoUrl,"POST",parmsData,function(result){
//			成功时关闭模态框
			if( result.errno == "1" ){
//				监控界面的模板
			    var templ =`
					<div class="modal fade task-modal" data-backdrop="static" data-show="true" id="modal-default">
						<div class="modal-dialog task-dialog" style="width:1000px;">
							<div class="modal-content task-modal-content">
								<div class="modal-header" style="height:50px;">
									<button type="button" class="close" title="关闭" onclick="main.modalClose()">
										<span aria-hidden="true" class="glyphicon glyphicon-remove"></span>
									</button>
									<h4 class="modal-title">信息展示</h4>
								</div>
								
			              		<div class="modal-body flowing-modal-body">
				              		<!-- 信息展示界面 -->
									<div style="height:100%;">
									    <div id="log-container">${result.data}</div>
									    ${buttomModel}
									</div>
			              		</div>
							</div>
						</div>
					</div>
			    `;
			    $('.testmodal').html(templ);
			    $('.testmodal #modal-default').modal('show');
			}else{
				console.log(result);
				util.errorView("系统异常");
		        return;
			}
		});
	},

//	添加bootstrap模板界面
	addModel: function(data,text){
	    var templ =`
			<div class="modal fade task-modal" id="modal-default">
				<div class="modal-dialog ftp-dialog">
					<div class="modal-content task-modal-content">
						<div class="modal-header">
		    				<button type="button" class="close" title="关闭" onclick="main.modalClose()">
		    					<span aria-hidden="true">&times;</span>
		    				</button>
							<h4 class="modal-title">模型${text}</h4>
						</div>
						<div class="modal-body task-modal-body">
						    <div class="right-content" style="margin:0 auto;width:350px;">
								<form id='taskForm2' class="form-horizontal">
			      					<div class="box-body">
										<!--遍历循环步骤一信息-->
						                ${data.map(f => `
											<div class="form-group ${f.class || ''}" style="${f.float || ''}">
												${f.html}
											</div>
						                `).join('')}
				      					<!-- 确定与取消按钮 -->
				      					<div class="box-footer">
							                <button type="button" class="btn btn-default" title="取消" onclick="main.modalClose()">取消</button>
							                <button type="button" class="btn btn-info pull-right" title="${text}">开始${text}</button>
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
	}
}

