
//点击GPU
$(".gpuInfo").click(function(){
	gpuInfo.Table();
	gpuInfo.activePage = 1;
})
//GPU管理
var gpuInfo = {
	
	activePage: 1,
	
	timeUpload: null,
	
	gpuList: {
		
	},
	
//	评估管理的模版列表
	Table: function(){
//	  	gpu任务列表的模板
	    var listModel = ` 
	    <div class="row">
		    <div class='testmodal'></div>
        	<div class="col-xs-12">
        		<div class="box box-primary">
        			<div class="box-header">
        				<div class="buttons"></div>
        				<h3 class="box-title">GPU信息</h3>
    				</div>
	    			<div class="box-body gpuBody">
	    			
	    			</div>
	    		</div>
	    	</div>
	    </div>
	    `;

	    $('.content.container-fluid').html(listModel);
	    gpuInfo.loadTable();
	    gpuInfo.timeUpload = setInterval(function(){
	    	gpuInfo.loadTable();
	    }, 3000);
	},
	
//	加载GPU的列表信息
	loadTable: function(){
//		使用get请求gpu
        $.ajax({
	        type : "get",
	        url : trainUrl+"/gups_info",
	        async : true,
	        data : {},
            success: function (data) {
            	var taskData1 = [];
            	for(var item in data.data){
            		for(var i=0; i<data.data[item].length; i++){
            			data.data[item][i].memoryutil = (parseInt( data.data[item][i].memoryutil*100 ))+"%";
            			data.data[item][i]["names"] = data.data[item][i]["pc-name"];
            			taskData1.push( data.data[item][i] );
            		}
            	}
            	timeTable(taskData1);
            }
       	});
       	function timeTable(taskData){
	        var tableModel = `
				<table id="example2" class="table table-bordered table-hover">
			        <thead>
				        <tr>
			                <th>pc-name</th>
			                <th>id</th>
			                <th>totalMemory</th>
			                <th>freeMemory</th>
			                <th>util</th>
			                <th>gpuUtil</th>
				        </tr>
			        </thead>
			        <tbody>
		                ${taskData.map(m => `
				            <tr>
				            	<td title="${m.names}">${m.names}</td>
				            	<td title="${m.id}">${m.id}</td>
				            	<td title="${m.totalMemory}">${m.totalMemory}</td>
				            	<td title="${m.freeMemory}">${m.freeMemory}</td>
				            	<td title="${m.memoryutil}">${m.memoryutil}</td>
				            	<td title="${m.gpuUtil}">${m.gpuUtil}</td>
				            </tr>
		                `).join('')}
			        </tbody>
				</table>
	        ` || "";
	        $('.content.container-fluid .box .gpuBody').html(tableModel);
	        
	        var table = $('#example2').DataTable({
				'language'    	: window.lang,
	            'paging'      	: true,
	            'lengthChange'	: false,
	            'searching'   	: false,
	            'ordering'    	: false,
	            'info'        	: true,
	            "order"			: [[1, "desc"]],
//		        'pagingType'  	: "simple",
//		        "lengthMenu"	: [15],
	            'autoWidth'   	: false
	        });
	        var pageVal = gpuInfo.activePage - 1;
	        $('#example2').DataTable().page(pageVal).draw(false);
//			列表点击分页时，之前选中的任务进行再次选中
		    $('#example2_paginate').on('click', function () {
				gpuInfo.activePage = $('#example2_paginate .paginate_button.active a').html();
		    });
       	}
		
	}
	
}
