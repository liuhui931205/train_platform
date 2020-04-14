
//点击Network
$(".modelList").click(function(){
	modelList.Table();
})
//Network管理
var modelList = {
	
//	Network的模版列表
	Table: function(){
		
//  	Network的模板
	    var listModel = ` 
	    <div class="row">
		    <div class='testmodal'></div>
        	<div class="col-xs-12">
        		<div class="box box-primary">
        			<div class="box-header">
        				<div class="buttons"></div>
        				<h3 class="box-title">识别模型</h3>
    				</div>
				    <div class="refresh" style="padding-left:10px;">
				      	<button type="button" onclick="modelList.Table()" class="btn btn-success" style="float:right;margin-right:10px;">
				      		列表刷新
				      	</button>
				    </div>
	    			<div class="box-body">
						<table id="example2" class="table table-bordered table-hover">
					        <thead>
						        <tr>
					                <th>adcode</th>
					                <th>modelEnv</th>
					                <th>modelName</th>
					                <th>modelDesc</th>
					                <th>modelVersion</th>
					                <th>cityName</th>
					                <th>modelDate</th>
						        </tr>
					        </thead>
		    			</table>
	    			</div>
	    		</div>
	    	</div>
	    </div>
	    `;

	    $('.content.container-fluid').html(listModel);

//		初始化表格(Network的属性)
    	var table = $("#example2").DataTable({
	    	'language'    : main.lang,
	        'searching'   : false,								//原生搜索
	        'paging'      : true,
	        'lengthChange': false,
	        'ordering'    : false,
	        'info'        : true,
	        'autoWidth'   : false,
	        "aaSorting"	  : false,
        	'processing'  : true,  								//隐藏加载提示,自行处理
        	'serverSide'  : false,  								//启用服务器端分页
	        ajax: function (data, callback, settings) {
	        	var param = {};
//				封装请求参数
	            param.limit = 10;								//页面显示记录条数，在页面显示每页显示多少项的时候
	            param.start = data.start;						//开始的记录序号
	            param.page = (data.start / data.length)+1;		//当前页码
			
//				使用post请求Network(每次只请求十条任务,然后对其进行分页)
	            $.ajax({
			        type : "get",
			        url : window.modelListUrl,
			        async : true,
			        data : {},
	                success: function (data) {
	                	var datas = [];
	                	for(var name in data.result){
	                		for(var adcode in data.result[name]){
								var thisChild = data.result[name][adcode];
								thisChild["adcode"] = adcode;
	                			datas.push( thisChild );
	                		}
	                	}
	                    var returnData = {};
	                    returnData.recordsTotal = datas.length;				//返回数据全部记录
	                    returnData.recordsFiltered = datas.length;			//后台不实现过滤功能，每次查询均视作全部结果
	                    returnData.data = datas;						//返回的数据列表
//						调用DataTables提供的callback方法，代表数据已封装完成并传回DataTables进行渲染
//						此时的数据需确保正确无误，异常判断应在执行此回调前自行处理完毕
                    	callback(returnData);
	                }
	           	});
	        },
//			Network的数据展示
        	columns: [
//      		第1列----adcode----显示
	            { 
	            	"data": "adcode" || "",
					"class": "adcode"
				},
//      		第2列----modelEnv----显示
	            { 
	            	"data": "modelEnv" || "",
					"class": "modelEnv"
				},
//      		第3列----modelName----显示
	            { 
	            	"data": "modelName" || "",
					"class": "modelName"
				},
//      		第4列----modelDesc----显示
	            { 
	            	"data": "modelDesc" || "",
					"class": "modelDesc"
				},
//      		第5列----modelVersion----显示
	            { 
	            	"data": "modelVersion" || "",
					"class": "modelVersion"
				},
//      		第6列----cityName----显示
	            { 
	            	"data": "cityName" || "",
					"class": "cityName"
				},
//      		第7列----modelDate----显示
	            { 
	            	"data": "modelDate" || "",
					"class": "modelDate"
				}
	        ]
	            
	    });
	}
	
	
}
