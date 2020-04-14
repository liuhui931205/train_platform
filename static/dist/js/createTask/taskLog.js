//创建任务的第四个环节--日志


var taskLog = {
	
	logTime: null,
	
	logModel: function(Model){
		var logUrl = trainUrl+"/output_log?task_id="+createTask.taskId;
		taskLog.logTime = setInterval(function(){
			util.newAjax(logUrl,"GET","",function(result){
				$("#log-container").html("");
				if( result.errno == 1 ){
	            	for(var i=0; i<result.data.length; i++){
	            		if(result.data[i]){
	            			var timeText = `<span>${result.data[i]}</span>`;
							$("#log-container").append(timeText);
							$("#log-container").scrollTop( $('#log-container')[0].scrollHeight );
	            		}
	            	}
				}else{
					util.errorView(result.message);
					return;
				}
			});
		},3000);
	}
	
}
