
//点击随机抽图
$(".randomImage").click(function(){
	randomImage.Table();
})
//随机抽图管理
var randomImage = {
	
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

//	随机抽图的模版列表
	Table: function(){
		
//  	随机抽图的模板
	    var listModel = ` 
	    <div class="row">
        	<div class="col-xs-12">
        		<div class="box box-primary">
        			<div class="box-header">
        				<div class="buttons"></div>
        				<h3 class="box-title">随机抽图</h3>
    				</div>
	    			<div class="box-body imgTaskBody">
		                ${randomImage.listForm.map(f => `
							<div class="form-group">
								<label for="${f.id}" class="col-sm-5 control-label">${f.name} :</label>
								<div class="col-sm-7">
	    							<input type="text" class="form-control" id="${f.id}" value="" placeholder="${f.text}" onfocus="this.placeholder=''" onblur="this.placeholder='${f.text}'">
								</div>
							</div>
		                `).join('')}
						<div class="form-group">
							<label title="是否打乱顺序" style="padding:5px 15px;">
								<input type="checkbox" id="isshuffle" value="true" />
								<span style="float:left;"></span>
								<b style="padding-left:5px;font-weight:normal;" title="是否打乱顺序">是否打乱顺序</b>
							</label>
						</div>
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
			randomImage.submitStart();
		})
	},
	
//	开始随机抽图功能,并带有进度
	submitStart: function(){
		var parmsData = {};
		for(var i=0; i<randomImage.listForm.length; i++){
			var id = randomImage.listForm[i].id;
			var value = $("#"+id).val();
			parmsData[id] = value;
		}
		parmsData["isshuffle"] = Number( $("#isshuffle")[0].checked );
		
        var submitUrl = trainUrl+"/sam_value";
        var processUrl = trainUrl+"/auto_sam";
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
