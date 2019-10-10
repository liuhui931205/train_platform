
var startTask = {
	
	startModel: function(){
		
		var formsData = [
			{
				"name" 	: "数据类型",
				"type" 	: "select",
				"id" 	: "data_type",
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
				"name" 	: "类别数目",
				"text"	: "类别数目",
				"type" 	: "input",
				"id" 	: "num_classes"
			},
			{
				"name" 	: "GPU",
				"type" 	: "input",
				"id" 	: "gpus",
				"title" : " (用逗号分割)"
			},
			{
				"name" 	: "批大小",
				"text"	: "批大小",
				"type" 	: "input",
				"id" 	: "batch_size"
			},
			{
				"name" 	: "全量迭代",
				"type" 	: "input",
				"text"	: "全量迭代",
				"id" 	: "num_epoch"
			},
			{
				"name" 	: "学习率调整步长",
				"type" 	: "input",
				"id" 	: "steps",
				"title" : " (用逗号分割)"
			},
			{
				"name" 	: "初始学习率",
				"text"	: "初始学习率",
				"type" 	: "input",
				"id" 	: "base_lr"
			},
			{
				"name" 	: "模型前缀",
				"text"	: "模型前缀",
				"type" 	: "input",
				"id" 	: "model"
			},
			{
				"name":"是否训练继续",
				"id":"w",
				"type":"checkbox",
				"event": true
			},
			{
				"name":"是否训练继续",
				"id":"weights",
				"type":"checkbox",
				"event": true
			},
			{
				"name" 	: "训练配置",
				"id" 	: "train_con",
				"type" 	: "text",
				"url":trainUrl+"/train_config?task_id="+createTask.taskId
			},
			{
				"name":"数据配置",
				"id":"data_con",
				"type":"text",
				"url":trainUrl+"/data_config?task_id="+createTask.taskId
			}
		];
		
		var newForm = myNetwork.format(formsData);
		createTask.stepForm.stepForm3 = newForm;
    	var testModel = `
			<!--遍历循环步骤三信息-->
            ${createTask.stepForm.stepForm3.map(f => `
				<div class="form-group ${f.id} ${f.display || ''}">
					${f.html}
				</div>
            `).join('')}
        `;
        $(".stepForm3").html(testModel);
	    
//		根据批大小获取全量迭代
	    $('#batch_size').blur(function () {
	        var batch_val = $(this).val();
			var url = trainUrl+"/calculate?batch_val="+batch_val+"&task_id="+createTask.taskId;
			var data = util.newAjax(url,"GET");
			if(data && data.data && (data.errno == '1')){
	            $('#num_epoch').val(data.data);
			}
	    })

		// $('#num_classes').blur(function () {
		//
		// 	var url = trainUrl+"/label_class?task_id="+createTask.taskId;
		// 	var data = util.newAjax(url,"GET");
		// 	if(data && data.data && (data.errno == '1')){
	    //         $('#num_epoch').val(data.data);
		// 	}
	    // })
	}
	
}
