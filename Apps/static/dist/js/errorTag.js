
//点击离线导入
$(".errorTag").click(function(){
	errorTag.Table();
})
//离线导入管理
var errorTag = {

	listForm: [
		{
			"name":"任务名",
			"text":"任务名",
			"id":"task_name"
		},
		{
			"name":"模型地址",
			"id":"weights_dir",
			"type":"select",
			"url":trainUrl+"/model_lists"
		},
		{
			"name":"gpu",
			"text":"gpu,以逗号分隔",
			"id":"gpus"
		}
	],

//	离线导入的模版列表
	Table: function(){
		
		var data = myNetwork.format(errorTag.listForm);
//  	离线导入的模板
	    var listModel = ` 
	    <div class="row">
        	<div class="col-xs-12">
        		<div class="box box-primary">
        			<div class="box-header">
        				<div class="buttons"></div>
        				<h3 class="box-title">检查项挑错</h3>
    				</div>
	    			<div class="box-body imgTaskBody">
		                ${data.map(d => `
							<div class="form-group">
								${d.html}
							</div>
		                `).join('')}
						<div style="height: 500px;">
							<label for="gpus" class="col-sm-5 control-label" style="text-align:left;">trackpointids :</label>
							<div class="col-sm-7">
								<textarea class="trackpointids" style="width: 300px;height: 500px;"></textarea>
							</div>
						</div>
						<div class="form-group">
	              			<button type="button" class="btn btn-success submitStart" style="margin-left:15px;">提交</button>
						</div>
	    			</div>
	    		</div>
	    	</div>
	    </div>
	    `;

	    $('.content.container-fluid').html(listModel);
	    
//		点击checkbox选中某种模型
		$(".submitStart").click(function(){
			errorTag.submitStart();
		})
	},
	
//	开始离线导入功能,并带有进度
	submitStart: function(){
		var parmsData = {};
		for(var i=0; i<errorTag.listForm.length; i++){
			var id = errorTag.listForm[i].id;
			var value = $("#"+id).val() || $("#"+id+" option:selected").val() || "";
			parmsData[id] = value;
		}
		var trackpointids = $(".trackpointids").val().split("\n").join(",");
		parmsData["trackpointids"] = trackpointids;
        var submitUrl = trainUrl+"/check_sele";
        var processUrl = trainUrl+"/check_data";
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
