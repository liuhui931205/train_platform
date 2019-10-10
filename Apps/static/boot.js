/**
 * map editor CSS and JS library
 */
(function() {
	
//	获取url配置文件( config配置文件，获取接口   )
	/*$.ajax( {
	    type : "get",
	    url : window._init_url,
	    async : false,
	    data : {},
	    success : function(data) {
//         	用户登录(OAuth)
			window.Kus = data.url.kus;
//         	OAuth登录
			window.Kd_auth_server = data.url.kd_auth_server;
//			任务系统
			window.kds_meta = data.url.kds_meta;
//			人员系统
			window.Kpms = data.url.kpms;
	    },
	   	error: function(XMLHttpRequest, textStatus, errorThrown) {
	   		console.log(XMLHttpRequest.status+"---"+XMLHttpRequest.readyState+"---"+textStatus)
	   	},
	});*/
	window.lang = {
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
	  	},
	};
	
	window.trainUrl = "http://10.11.5.170:9000/api";
	window.modelListUrl = "http://10.11.5.121:9527/getRecogModel";
	
//	左侧选项卡切换
	$(".treeview-menu li").click(function(){
		if($(".treeview-menu li.active").length>0){
			$(".treeview-menu li.active").removeClass("active");
		}
		if($(".sidebar-menu li.active").length>0){
			$(".sidebar-menu li.active").removeClass("active");
		}
		$(this).addClass("active");
		$(this).parent().parent().addClass("active");
		clearInterval(gpuInfo.timeUpload);
	});
	
})();