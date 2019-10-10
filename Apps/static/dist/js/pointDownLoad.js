
//点击轨迹点下载
$(".pointDownLoad").click(function(){
	pointDownLoad.Table();
})
//轨迹点下载管理
var pointDownLoad = {

	listForm: [
		{
			"name":"dir_name",
			"text":"dir_name",
			"id":"dir_name"
		}
	],

//	轨迹点下载的模版列表
	Table: function(){
		
		var data = myNetwork.format(pointDownLoad.listForm);
//  	轨迹点下载的模板
	    var listModel = ` 
	    <div class="row">
        	<div class="col-xs-12">
        		<div class="box box-primary">
        			<div class="box-header">
        				<div class="buttons"></div>
        				<h3 class="box-title">轨迹点下载</h3>
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
			pointDownLoad.submitStart();
		})
	},
	
//	轨迹点下载功能
	submitStart: function(){
		var parmsData = {};
		for(var i=0; i<pointDownLoad.listForm.length; i++){
			var id = pointDownLoad.listForm[i].id;
			var value = $("#"+id).val() || $("#"+id+" option:selected").val() || "";
			parmsData[id] = value;
		}
		var trackpointids = $(".trackpointids").val();
		parmsData["trackpointids"] = trackpointids;
        var submitUrl = trainUrl+"/track_down";
		util.newAjax(submitUrl,"POST",parmsData,function(result){
//			成功与失败事件
			if( result.errno == 1 ){
				util.errorView(result.message,true);
				pointDownLoad.Table();
			}else{
				console.log(result);
				util.errorView(result.message);
		        return;
			}
		});
	}
}
