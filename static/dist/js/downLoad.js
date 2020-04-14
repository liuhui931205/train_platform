
//点击在线下载
$(".downLoad").click(function(){
	downLoad.Table();
})
//在线下载管理
var downLoad = {
	
	l_value: "",
	
	listForm: [
		{
			"name": "起始packid",
			"id": "taskid_start",
			"text": "起始packid"
		},
		{
			"name": "结束packid",
			"id": "taskid_end",
			"text": "结束packid"
		},
		{
			"name": "NAME",
			"id": "dest",
			"text": "NAME"
		}
	],

//	在线下载的模版列表
	Table: function(){
		
//  	在线下载的模板
	    var listModel = ` 
	    <div class="row">
        	<div class="col-xs-12">
        		<div class="box box-primary">
        			<div class="box-header">
        				<div class="buttons"></div>
        				<h3 class="box-title">在线下载</h3>
    				</div>
	    			<div class="box-body imgTaskBody">
		                ${downLoad.listForm.map(f => `
							<div class="form-group">
								<label for="${f.id}" class="col-sm-5 control-label">${f.name} :</label>
								<div class="col-sm-7">
	    							<input type="text" class="form-control" id="${f.id}" value="" placeholder="${f.text}" onfocus="this.placeholder=''" onblur="this.placeholder='${f.text}'">
								</div>
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
			downLoad.submitStart();
		})
	},
	
//	开始在线下载功能,并带有进度
	submitStart: function(){
		var parmsData = {};
		for(var i=0; i<downLoad.listForm.length; i++){
			var id = downLoad.listForm[i].id;
			var value = $("#"+id).val();
			parmsData[id] = value;
		}
//		下载并进行评估
		/*parmsData["l_value"] = $("#l_value")[0].checked || "";
		downLoad.l_value = parmsData["l_value"];*/
        var submitUrl = trainUrl+"/linedown";
        var processUrl = trainUrl+"/line_status";
		util.newAjax(submitUrl,"POST",parmsData,function(result){
//			成功时进行进度条转换( resp.errno=='1'的时候 )
//			否则为发布失败   -- 进行提示
			if( result.errno == 1 ){
				util.errorView(result.message,true);
				main.modalClose();
				createTask.taskProgress(result.data.task_id, processUrl, function(){
//					跳转模型
// 					if(downLoad.l_value){
// 						$('.t-model>a').click();
// 						$('.myModel>a').click();
// 					}
				});
			}else{
				console.log(result);
				util.errorView(result.message);
		        return;
			}
		});
	}
}
