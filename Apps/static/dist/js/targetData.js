
//点击target数据生成
$(".targetData").click(function(){
	targetData.Table();
})
//target数据生成管理
var targetData = {

	listForm: [
		{
			"name":"version",
			"text":"version",
			"id":"version"
		},
		{
			"name":"step",
			"text":"step",
			"id":"step"
		}
	],

//	target数据生成的模版列表
	Table: function(){
		
		var data = myNetwork.format(targetData.listForm);
//  	target数据生成的模板
	    var listModel = ` 
	    <div class="row">
        	<div class="col-xs-12">
        		<div class="box box-primary">
        			<div class="box-header">
        				<div class="buttons"></div>
        				<h3 class="box-title">target数据生成</h3>
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
	    
//		点击开始执行按钮
		$(".submitStart").click(function(){
			targetData.submitStart();
		})
	},
	
//	开始任务包生成功能,并带有进度
	submitStart: function(){
		var parmsData = {};
		for(var i=0; i<targetData.listForm.length; i++){
			var id = targetData.listForm[i].id;
			var value = $("#"+id).val();
			if(!value){
				util.errorView(id+"参数不能为空");
				return;
			}
			parmsData[id] = value;
		}
		
        var submitUrl = trainUrl+"/target_divide";
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
