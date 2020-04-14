
//点击离线导入
$(".imports").click(function(){
	imports.Table();
})
//离线导入管理
var imports = {

	listForm: [
		{
			"name":"色板信息",
			"text":"色板信息",
			"id":"roadelement"
		},
		{
			"name":"数据来源",
			"id":"source",
			"type":"select",
			"data":[
				{
					"name":"在线导入",
					"value":"0"
				},
				{
					"name":"离线导入",
					"value":"1"
				},
				{
					"name":"人工挑图",
					"value":"2"
				}
			]
		},
		{
			"name":"作者信息",
			"text":"作者信息",
			"id":"author"
		},
		{
			"name":"标注类型",
			"id":"annotype",
			"type":"select",
			"data":[
				{
					"name":"数据标注",
					"value":"0"
				},
				{
					"name":"返工标注",
					"value":"1"
				},
				{
					"name":"二次返工",
					"value":"2"
				}
			]
		},
		{
			"name":"数据种类",
			"id":"datakind",
			"type":"select",
			"data":[
				{
					"name":"正式数据",
					"value":"0"
				},
				{
					"name":"测试数据",
					"value":"1"
				}
			]
		},
		{
			"name":"城市",
			"text":"城市",
			"id":"city"
		},
		{
			"name":"NAME",
			"text":"NAME",
			"id":"src"
		},
		{
			"name":"图片操作范围",
			"id":"imgoprange",
			"type":"select",
			"data":[
				{
					"name":"半图",
					"value":"0"
				},
				{
					"name":"全图",
					"value":"1"
				}
			]
		}
	],

//	离线导入的模版列表
	Table: function(){
		
		var data = myNetwork.format(imports.listForm);
//  	离线导入的模板
	    var listModel = ` 
	    <div class="row">
        	<div class="col-xs-12">
        		<div class="box box-primary">
        			<div class="box-header">
        				<div class="buttons"></div>
        				<h3 class="box-title">离线导入</h3>
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
			imports.submitStart();
		})
	},
	
//	开始离线导入功能,并带有进度
	submitStart: function(){
		var parmsData = {};
		for(var i=0; i<imports.listForm.length; i++){
			var id = imports.listForm[i].id;
			var value = $("#"+id).val() || $("#"+id+" option:selected").val() || "";
			parmsData[id] = value;
		}
		
        var submitUrl = trainUrl+"/offimport";
        var processUrl = trainUrl+"/off_status";
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
