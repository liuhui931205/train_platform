
//点击评估
$(".assessment").click(function(){
	assessment.Table();
})
//评估管理
var assessment = {
	
	dir: "",				//未刷新页面缓存图片地址
	
	dest_dir: "",			//未刷新页面缓存图片地址
	
	page: 1,
	
	total_img: null,
	
//	评估管理的模版列表
	Table: function(){
		var parmsData = {
			"task_id": myModel.taskId,
			"cur_img": assessment.page
		}
		var url = trainUrl+"/show";
		util.newAjax(url,"POST",parmsData,function(result){
			if(result.errno == "1"){
				assessment.total_img = result.total_img;
				assessment.imageLoading(result);
			}else{
				assessment.page = 0;
				util.errorView("找不到图片");
			}
		});
//  	评估的模板
	    var listModel = ` 
	    <div class="row">
        	<div class="col-xs-12">
        		<div class="box box-primary">
        			<div class="box-header">
        				<div class="buttons"></div>
        				<h3 class="box-title">评估</h3>
    				</div>
	    			<div class="box-body-right">
	    				<div class="userImage image1">
							<img src="">
							<p>原图</p>
	    				</div>
	    				<div class="userImage image2">
							<img src="">
							<p></p>
	    				</div>
	    				<div class="userImage image3">
							<img src="">
							<p></p>
	    				</div>
	    				<div class="userImage image4">
							<img src="">
							<p></p>
	    				</div>
	    			</div>
	    			<div class="box-body-left">
	              		<button type="button" class="btn btn-success lastImage">上一张</button>
	              		<button type="button" class="btn btn-success nextImage">下一张</button>
						<div class="allNumber">
				      		<input type="number" class="form-control beforeIndex" value="0">
				      		<span class="allIndex"> / 0</span>
				      	</div>
	    			</div>
	    		</div>
	    	</div>
	    </div>
	    `;
	    $('.content.container-fluid').html(listModel);
//		上一张图片
	    $(".lastImage").click(function(){
	    	var index = Number( $(".beforeIndex").val() );
	    	if(assessment.page && (index>1)){
	    		assessment.page--;
	    		assessment.Table();
	    	}
	    })
//		下一张图片
	    $(".nextImage").click(function(){
	    	var index = Number( $(".beforeIndex").val() );
	    	if(assessment.page && (index<assessment.total_img)){
	    		assessment.page++;
	    		assessment.Table();
	    	}
	    })
//		根据输入框跳转对应帧图片--回车
	    $(".beforeIndex").keydown(function(event){
	    	assessment.jumpImage(event);
	    })
//		根据输入框跳转对应帧图片--回车
	    $(".beforeIndex").blur(function(event){
	    	assessment.jumpImage(event);
	    })
	},
	
	jumpImage: function(event){
//		判断是否为数字
		var beforeNumber = Number( $(".beforeIndex").val() );
    	var patrn = /^[0-9]*$/;
	    if (patrn.exec(beforeNumber) == null || beforeNumber == "") {
	        return;
	    }
//		执行跳转
		var ev = document.all ? window.event : event;
		if( (ev.keyCode == 13) || (ev.type == "blur") ) {
	        assessment.page = beforeNumber;
//			执行修改图片src地址的事件
	    	assessment.Table();
		}
	},
	
//	加载评估预测的图片
	imageLoading: function(result){
//		修改图片的地址
		var originalImage = result.or_data;
		var recognitionImage = result.la_data;
	    $('.image1 img').attr('src', 'data:image/jpg;base64,' + originalImage);
	    for(var i=0; i<recognitionImage.length; i++){
	    	var imageIndex = i+2,
	    		modelName = recognitionImage[i].name,
	    		imageUrl = recognitionImage[i].data;
	    	$('.image'+imageIndex+' img').attr('src', 'data:image/png;base64,' + imageUrl);
	    	$('.image'+imageIndex+' p').html(modelName);
	    }
//	    $('.image2 img').attr('src', 'data:image/png;base64,' + recognitionImage);
//		修改与图片相对应的队列
	    var index = assessment.page;
	    var allIndex = " / "+assessment.total_img;
	    $(".beforeIndex").val(index);
	    $(".allIndex").html(allIndex);
	    
	}
}
