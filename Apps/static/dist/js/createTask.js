

var createTask = {
	
	logTime: null,
	
	taskId: null,
	
	taskName: null,
	
	dataName: null,
	
	model: null,
	
	submitUrl: [{
		"url" : trainUrl+"/create_train",
		"name" : "保存"
	},{
		"url" : trainUrl+"/save_map",
		"name" : "开始训练"
	},{
		"url" : trainUrl+"/starttask",
		"name" : "历史任务"
	}],
	
//	所需要填写的参数
	stepForm: {
		"stepForm1": [
			{
				"name" 	: "训练任务名称",
				"text"	: "训练任务名称",
				"type" 	: "input",
				"id" 	: "train_name"
			},
			{
					"name" 	: "服务器",
					"text"	: "服务器",
					"type" 	: "input",
					"id" 	: "host"
			},
			{
					"name":"图片类型",
					"id":"image_type",
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
				"name" 	: "选择network",
				"type" 	: "select",
				"id" 	: "network",
				"url"	: trainUrl+"/network"
			},
			{
				"name" 	: "选择data",
				"type" 	: "select",
				"id" 	: "data",
				"url"	: trainUrl+"/sour_datas"
			},
			{
				"name" 	: "任务类型",
				"type" 	: "select",
				"id" 	: "types",
				"url"	: trainUrl+"/task_type"
			},
			{
				"name" 	: "映射模板",
				"type" 	: "select",
				"id" 	: "maps",
				"url"	: trainUrl+"/map_template"
			},
			{
				"name" 	: "task描述",
				"type" 	: "text",
				"id" 	: "task_desc"
			},
			{
					"name":"并行训练",
					"type":"checkbox",
					"id":"parallel_bool"
			},
		],
		
		"stepForm2":[],
		"stepForm3":[]
	},
	
	thisStep: 0,
	
//	创建训练任务模板
	createModel: function(){
		var checkArr = [],
			headers = "创建训练任务";
		for(var item in trainTask.checkTask){
			if(trainTask.checkTask[item]){
				checkArr.push( trainTask.checkTask[item] );
			}
		}
		if(checkArr.length == 1){
			util.errorView("已选择1条历史任务-优化训练任务");
			headers = "优化训练任务";
		}else{
			util.errorView("已选择"+checkArr.length+"条历史任务-创建训练任务",true);
		}
		createTask.stepForm = {
			"stepForm1": [
				{
					"name" 	: "训练任务名称",
					"text"	: "训练任务名称",
					"type" 	: "input",
					"id" 	: "train_name"
				},
				{
					"name" 	: "服务器",
					"text"	: "服务器",
					"type" 	: "input",
					"id" 	: "host"
				},
				{
					"name":"图片类型",
					"id":"image_type",
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
					"name" 	: "选择network",
					"type" 	: "select",
					"id" 	: "network",
					"url"	: trainUrl+"/network"
				},
				{
					"name" 	: "选择data",
					"type" 	: "select",
					"id" 	: "data",
					"url"	: trainUrl+"/sour_datas"
				},
				{
					"name" 	: "任务类型",
					"type" 	: "select",
					"id" 	: "types",
					"url"	: trainUrl+"/task_type"
				},
				{
				"name" 	: "映射模板",
				"type" 	: "select",
				"id" 	: "maps",
				"url"	: trainUrl+"/map_template"
				},
				{
					"name" 	: "task描述",
					"type" 	: "text",
					"id" 	: "task_desc"
				},
				{
					"name":"并行训练",
					"type":"checkbox",
					"id":"parallel_bool"
			},
			],
			
			"stepForm2":[],
			"stepForm3":[]
		};
//		监控界面的模板
	    for(var item in createTask.stepForm){
	    	var forms = createTask.stepForm[item];
			createTask.stepForm[item] = myNetwork.format(forms);
	    }
	    
	    var templ =`
			<div class="modal fade createModel" data-backdrop="static" data-show="true" id="modal-default">
				<div class="modal-dialog task-dialog">
					<div class="modal-content task-modal-content">
						<div class="modal-header" style="height:6%;">
							<button type="button" class="close" title="关闭" onclick="createTask.modalClose()">
								<span aria-hidden="true" class="glyphicon glyphicon-remove"></span>
							</button>
							<h4 class="modal-title">${headers}</h4>
						</div>
	              		<div class="modal-body" style="height:94%;padding:0px;">
	              			<button type="button" class="btn btn-success nextStep">创建训练任务</button>
		              		<!-- 监控界面 -->
							<div class="stepForm1 opacity" title="stepForm1">
								<!--遍历循环步骤一信息-->
				                ${createTask.stepForm.stepForm1.map(f => `
									<div class="form-group">
										${f.html}
									</div>
				                `).join('')}
							</div>
							<div class="stepForm2" title="stepForm2">
							
							</div>
							<div class="stepForm3" title="stepForm3">
							
							</div>
							<div class="stepForm4">
			              		<!-- 监控界面 -->
								<div style="height:95%;">
			              			<!-- websocket日志界面 -->
								    <div id="log-container" style="height:100% !important;"></div>
								    <button type="button" class="btn btn-default endTask" title="结束训练" onclick="createTask.endTask()">结束训练</button>
								</div>
								
							</div>
	              		</div>
					</div>
				</div>
			</div>
	    `;
	    $('.testmodal').html(templ);
//	    $('#modal-default').modal('show');
	    
//		点击进入下一步的按钮
	    $('.nextStep').click(function () {
	    	createTask.stepOpacity(checkArr);
	    })
	    
//		点击模态框背景时
		$(".modal-backdrop.fade.in").click(function(){
			clearInterval(taskLog.logTime);
		})
	},
	
//	创建训练任务的关闭
	modalClose: function(){
		main.modalClose();
		createTask.thisStep = 0;
	},
	
//	点击下一步对其他模版进行遮挡不能操作
	stepOpacity: function(checkArr){
		if(createTask.thisStep > 2){
			main.modalClose();
			createTask.thisStep = 0;
			return;
		}
		
		createTask.netSubmit(createTask.thisStep, checkArr);
	},
	
//	每次点下一步时提交的参数
	netSubmit: function(num, checkArr){
		var domName = $(".createModel .modal-body>div")[num].title;
		var formDom = createTask.stepForm[domName];
		var jsonObj = {},
			map_value = {};
		for(var i=0; i<formDom.length; i++){
			var id = formDom[i].id;
			var key = formDom[i].id;
			if( (id=="w") && ($("#"+id)[0].checked) ){
				key = "weights";
				var value = $("#weights option:selected").val() || "";
			}else if( (id=="w") && (!$("#"+id)[0].checked) ){
				key = "weights";
				var value = "";
			}else if( $("#"+id).length ){
				var value = $("#"+id).val() || $("#"+id+" option:selected").val() || $("#"+id)[0].checked || "";
				// var value = $("#"+id).val() || $("#"+id+" option:selected").val() || "";
			}
			if(num == 1){
				map_value[key] = [
					formDom[i].value,
					formDom[i].class,
					formDom[i].name,
					value
				];
			}else{
				jsonObj[key] = value;
			}
		}
		if(num == 1){
			jsonObj["map_value"] = map_value;
			jsonObj["task_id"] = createTask.taskId;
		}else if(num == 2){
			var data = util.Time( (new Date()),true );
			jsonObj["start_date"] = data;
			jsonObj["task_id"] = createTask.taskId;
			createTask.model = jsonObj["model"];
		}else if( (num == 0) && (checkArr.length == 1) ){
			jsonObj["task_id"] = checkArr[0];
		}
		if(jsonObj["train_name"] && jsonObj["data"]){
			createTask.taskName = jsonObj["train_name"];
			createTask.dataName = jsonObj["data"];
		}

		var url = createTask.submitUrl[num].url;
		util.newAjax(url, "POST", jsonObj, function(result){
			var view = result.resp || result.message;
		    util.errorView(view);
			if( (num == 0) && (result.errno == 1) ){
				createTask.taskId = result.data.task_id;
				createMap.gainData(createTask.taskId);
			}else if( (num == 1) && (result.errno == 1) ){
				var view = result.resp || result.errmsg;
				startTask.startModel();
			}else if( (num == 2) && (result.errno == 1) ){
				var view = result.resp || result.errmsg;
				taskLog.logModel(jsonObj["model"]);
			}else if( result.errno != 1 ){
		        return;
			}
			if( result.errno == 1 ){
				createTask.success(num);
			}
		});
	},
	
//	进入下一步的函数执行成功
	success: function(num){
		createTask.thisStep += 1;
		$(".createModel .modal-body>div.opacity").removeClass("opacity");
		var doms = $(".createModel .modal-body>div");
		for(var i=0; i<doms.length; i++){
			if(i == createTask.thisStep){
				$(".createModel .modal-body>div").eq(i).addClass("opacity");
			}
		}
		var buttonName = createTask.submitUrl[num].name;
		$('.nextStep').html(buttonName);
	},
	
//	是否继续训练的checkbox选项
	Continue: function(judge,id){
		if(judge){
			var addClass = $("#"+id)[0].checked;
			if(addClass){
				var formArr = [];
				var weightUrl = trainUrl+"/weights?task_id="+createTask.taskId;
				var getData = util.newAjax(weightUrl,"GET") || {"data":[]};
				for(var l=0; l<getData.data.length; l++){
					formArr.push({
						"name": getData.data[l],
						"value": getData.data[l]
					})
				}
				
				var doms = `
					<label for="weights" class="col-sm-5 control-label" style="text-align:left;">是否继续训练 :</label>
					<div class="col-sm-7">
						<select class="form-control" id="weights" title="是否继续训练">
				            <option class="change" value="">是否继续训练</option>
			                ${formArr.map(d => `
				            	<option class="change" value="${d.value}">${d.name}</option>
			                `).join('')}
			            </select>
					</div>
				`;
				$(".weights").html(doms);
				$(".weights").css("display","inline-block");
			}else{
				$(".weights select").remove();
				$(".weights").css("display","none");
			}
		}
	},
	
//	结束训练
	endTask: function(){
		clearInterval(taskLog.logTime);
		var end_date = util.Time( (new Date()),true );
        var params = {
            'end_date': end_date,
            'task_id': createTask.taskId
        }
        var url = trainUrl+"/endtask";
		util.newAjax(url,"POST",params,function(result){
			if(result.errno == 1){
				util.errorView(result.message);
			}
		});
	},
	
//	进度条信息加载
	taskProgress: function(taskId,url,callback=false){
		$("#progressing .progress-bar").css("width","0");
    	$("#progressing").css("display","block");
		var dataForm = {
			"task_id" : taskId
		}
		createTask.logTime = setInterval(function(){
			util.newAjax(url,"POST",dataForm,function(result){
				if(result.errno == 1){
					if(result.data[0].status == "completed!"){
						if(callback){
							clearTime();
							callback();
						}else{
							clearTime();
						}
					}else{
						var lodNum = result.data[0].status;
						$("#progressing .progress-bar").css("width",lodNum);
					}
				}else{
					clearTime();
					return;
				}
			});
		},5000);
		
		function clearTime(){
			$("#progressing").css("display","none");
			clearInterval(createTask.logTime);
			main.modalClose();
		}
		
	}
	
	
}
