
//点击任务包生成
$(".taskCreate").click(function(){
	taskCreate.Table();
})
//任务包生成管理
var taskCreate = {

	listForm: [
		{
			"name":"文件夹名",
			"text":"文件夹名",
			"id":"version"
		},
		{
			"name":"step",
			"text":"step",
			"id":"step"
		},
		{
			"name":"文件类型",
			"id":"types",
			"type":"select",
			"data":[
				{
					"name":"全图",
					"value":"full"
				},
				{
					"name":"半图",
					"value":"remote"
				}
			]
		}
	],

//	任务包生成的模版列表
	Table: function(){
		
		var data = myNetwork.format(taskCreate.listForm);
//  	任务包生成的模板
	    var listModel = ` 
	    <div class="row">
        	<div class="col-xs-12">
        		<div class="box box-primary">
        			<div class="box-header">
        				<div class="buttons"></div>
        				<h3 class="box-title">任务包生成</h3>
    				</div>
	    			<div class="box-body imgTaskBody">
		                ${data.map(d => `
							<div class="form-group">
								${d.html}
							</div>
		                `).join('')}
						<div class="form-group">
	              			<button type="button" class="btn btn-success submitStart" style="margin-left:15px;">开始</button>
						</div>
	    			</div>
	    		</div>
	    	</div>
	    </div>
	    `;

	    $('.content.container-fluid').html(listModel);
	    
//		点击checkbox选中某种模型
		$(".submitStart").click(function(){
			taskCreate.submitStart();
		})
	},
	
//	开始任务包生成功能,并带有进度
	submitStart: function(){
		var parmsData = {};
		for(var i=0; i<taskCreate.listForm.length; i++){
			var id = taskCreate.listForm[i].id;
			var value = $("#"+id).val();
			parmsData[id] = value;
		}
		
        var submitUrl = trainUrl+"/task_divide";
        var processUrl = trainUrl+"/divide_history";
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
	}
}
