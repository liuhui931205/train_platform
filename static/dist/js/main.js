

var main = {
	
//	导出库功能
	export: function(value){
		console.log(value);
	},
	
//	关闭模态框
	modalClose: function(){
		clearInterval(taskLog.logTime);
		$("#log-container").html("");
	    $('#modal-default').modal('hide');
	    $('.modal-backdrop').remove();
	},
	
//	过滤条件搜索功能
	searchType: function(event,Library){
		if(event.keyCode == 13) {
			var searchType = $('.searchType option:selected').val();	//搜索类型
			var searchValue = $(".searchTypeValue").val();				//搜索条件
			
			console.log(searchType+" : "+searchValue);
			console.log(event);
//			console.log( Library.Table );
		}
	},
	
//-----------------------------------------列表的中文显示----------------------------------------
	lang: {
	    "sProcessing": "加载中...",
		"sLengthMenu": "显示 _MENU_ 项结果",  
		"sZeroRecords": "没有匹配结果",  
		"sInfo": "第 _START_ 至 _END_ 项结果，共 _TOTAL_ 项",  
	  	"sInfoEmpty": "第 0 至 0 项结果，共 0 项",  
	  	"sInfoFiltered": "(由 _MAX_ 项结果过滤)",  
	  	"sInfoPostFix": "",  
	  	"sSearch": "搜索:",  
	  	"sUrl": "",  
	  	"sEmptyTable": "空",  
	  	"sLoadingRecords": "载入中...",  
	  	"sInfoThousands": ",",  
	  	"oPaginate": {
	      	"sFirst": "首页",  
	      	"sPrevious": "上页",  
	      	"sNext": "下页",  
	      	"sLast": "末页"  
	  	}
	}
	
	
}
