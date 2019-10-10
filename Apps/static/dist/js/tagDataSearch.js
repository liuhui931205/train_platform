
//点击标注数据查询
$(".tagDataSearch").click(function(){
	tagDataSearch.searchList = {
		"imgrange": 0,
		"label_info": "",
		"pacid": "",
		"tag_info": "",
		"city": ""
	};
	if(!tagDataSearch.classInfos.length){
		$("#loading").css("display","inline-block");
		tagDataSearch.class_infos();
	}else{
		tagDataSearch.Table();
	}
})
//标注数据查询管理
var tagDataSearch = {
	
	labelInfos: [],
	
	classInfos: [],
	
	searchList:{
		"imgrange": 0,
		"label_info": "",
		"pacid": "",
		"tag_info": "",
		"city": "",
		"pag_id":""
	},
	
//	标注数据查询的模版列表
	Table: function(){
		$("#loading").css("display","none");
//  	标注数据查询的模板
	    var listModel = ` 
	    <div class="row">
		    <div class='testmodal'></div>
        	<div class="col-xs-12">
        		<div class="box box-primary">
        			<div class="box-header">
        				<div class="buttons"></div>
        				<h3 class="box-title">标注数据查询</h3>
    				</div>
			        <div class="searchType" style="float:left;margin-left:10px;padding:5px;border:1px solid #3c8dbc;">
						<select class="form-control imgrange" style="width:100px;float:left;margin-right:5px;">
				            <option value="0">半图</option>
				            <option value="1">全图</option>
			            </select>
						<select class="form-control label_info" style="width:150px;float:left;margin-right:5px;">
			                <option value="">请选择label_info</option>
			                ${tagDataSearch.classInfos.map(c => `
			                	<option value="${c}">${c}</option>
			                `).join('')}
			            </select>
			            <div style="width:200px;float:left;margin-right:5px;">
							<select class="form-control tag_info selectpicker" multiple>
				                <option value="">请选择tag_info</option>
				                ${tagDataSearch.labelInfos.map(c => `
				                	<option value="${c.value}">${c.name}-${c.value}</option>
				                `).join('')}
				            </select>
			            </div>
			            <input type="text" class="form-control city" style="float:left;width:100px;margin-right:5px;" placeholder="任务id" />
	            		<input type="text" class="form-control city" style="float:left;width:100px;margin-right:5px;" placeholder="城市" />
						<button type="button" class="btn btn-success" onclick="tagDataSearch.search_table()">搜索</button>
			        </div>
				    <div class="refresh" style="padding-left:10px;">
				      	<button type="button" onclick="tagDataSearch.Table_s()" class="btn btn-success" style="float:right;margin-right:10px;">
				      		列表刷新
				      	</button>
				    </div>
	    			<div class="box-body">
	    			
	    			</div>
	    		</div>
	    	</div>
	    </div>
	    `;

	    $('.content.container-fluid').html(listModel);
		$('.selectpicker').selectpicker({
            'selectedText':'cat',
            'showSubtext':true
        })
	    tagDataSearch.Table_s();
	},
	
//	只刷新列表数据
	Table_s: function(){
		$(".box-body").html(`
			<table id="example2" class="table table-bordered table-hover">
		        <thead>
			        <tr>
		                <th>imgrange</th>
		                <th>pacid</th>
		                <th>city</th>
		                <th>label_info</th>
		                <th>time_info</th>
		                <th>trackpointid</th>
		                <th>tag_info</th>
			        </tr>
		        </thead>
			</table>
		`);
//		初始化表格(标注数据查询的属性)
    	var table = $("#example2").DataTable({
	    	'language'    : main.lang,
	        'searching'   : false,								//原生搜索
	        'paging'      : true,
	        'lengthChange': false,
	        'ordering'    : false,
	        'info'        : true,
	        'autoWidth'   : false,
	        "aaSorting"	  : false,
	        'scrollY'		: 600,
	        'scrollCollapse': true,
	        "lengthMenu"	: [15],
        	'processing'  : true,  								//隐藏加载提示,自行处理
        	'serverSide'  : true,  								//启用服务器端分页
	        ajax: function (data, callback, settings) {
	        	var param = {};
//				封装请求参数
	            param.limit = 10;								//页面显示记录条数，在页面显示每页显示多少项的时候
	            param.start = data.start;						//开始的记录序号
	            param.page = (data.start / data.length)+1;		//当前页码
				var post_data = tagDataSearch.searchList;
				post_data["pag_id"] = param.page;
//				使用post请求标注数据查询(每次只请求十条任务,然后对其进行分页)
	            $.ajax({
			        type : "post",
			        url : window.trainUrl+"/query_data",
			        async : true,
					contentType: "application/json; charset=utf-8",
			        data : JSON.stringify(post_data),
	                success: function (data) {
	                	var datas = data.data;
	                	/*for(var name in data.result){
	                		for(var adcode in data.result[name]){
								var thisChild = data.result[name][adcode];
								thisChild["adcode"] = adcode;
	                			datas.push( thisChild );
	                		}
	                	}*/
	                    var returnData = {};
	                    returnData.recordsTotal = data.count;				//返回数据全部记录
	                    returnData.recordsFiltered = data.count;			//后台不实现过滤功能，每次查询均视作全部结果
	                    returnData.data = datas;						//返回的数据列表
//						调用DataTables提供的callback方法，代表数据已封装完成并传回DataTables进行渲染
//						此时的数据需确保正确无误，异常判断应在执行此回调前自行处理完毕
                    	callback(returnData);
	                }
	           	});
	        },
//			标注数据查询的数据展示
        	columns: [
//      		第1列----adcode----显示
	            { 
	            	"data": "imgrange" || "",
					"class": "_imgrange"
				},
				//第2列----modelEnv----显示
	            {
	            	"data": "pacid" || "",
					"class": "_pacid"
				},
//      		第2列----modelEnv----显示
	            { 
	            	"data": "city" || "",
					"class": "_city"
				},
//      		第3列----modelName----显示
	            { 
	            	"data": "label_info" || "",
					"class": "_label_info"
				},
//      		第4列----modelDesc----显示
	            { 
	            	"data": "time_info" || "",
					"class": "_time_info"
				},
//      		第5列----modelVersion----显示
	            { 
	            	"data": "trackpointid" || "",
					"class": "_trackpointid"
				},
//      		第6列----cityName----显示
	            { 
	            	"data": "tag_info" || "",
	              	"render": function ( data, type, row,  meta ) {
		    			var status_html = `
              				<span title='${data}'>${data}</span>
              			`;
                        return status_html;
					},
					"class": "_tag_info"
				}
	        ]
	            
	    });
	},
	
//	条件查询
	search_table: function(){
		for(var item in tagDataSearch.searchList){
			var _value = $("."+item+" option:selected").val() || $("."+item).val() || "";
			if(item == "tag_info"){
				_value = $("."+item).val();
			}
			tagDataSearch.searchList[item] = _value;
		}
		tagDataSearch.Table_s();
	},
	
//	label_info类型查询
	class_infos: function(){
		var class_infos_url = window.trainUrl+"/class_info";
		util.newAjax(class_infos_url,"GET","",function(data){
			if( data.errno == 1 ){
				tagDataSearch.classInfos = data.data;
				if(!tagDataSearch.labelInfos.length){
					tagDataSearch.label_infos();
				}else{
					tagDataSearch.Table();
				}
			}else{
				console.log(data);
				util.errorView(data.message);
		        return;
			}
		});
	},
	
//	tag_info类型查询
	label_infos: function(){
		var label_infos_url = window.trainUrl+"/label_infos";
		util.newAjax(label_infos_url,"GET","",function(data){
			if( data.errno == 1 ){
				tagDataSearch.labelInfos = [];
				for(var item in data.data){
					tagDataSearch.labelInfos.push({
						"name": item,
						"value": data.data[item]
					})
				}
				tagDataSearch.Table();
			}else{
				console.log(data);
				util.errorView(data.message);
		        return;
			}
		});
	}
	
	
}
