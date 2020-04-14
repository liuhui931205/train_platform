
var createMap = {

	formValue: [],

    gainData: function (taskId) {
        var mapsUrl = trainUrl + "/seg_maps?task_id=" + taskId;
        util.newAjax(mapsUrl, "GET", [], function (data) {
            createMap.formValue = data.data;
            createMap.getFormValue(taskId)
        });
    },
	
//	获取第二步骤所需要填写的参数
	getFormValue: function(taskId){
		var gainUrl = trainUrl+"/maps?taskName="+createTask.taskName;
		var parms = {
			"task_id":taskId
		}
		var data = util.newAjax(gainUrl,"POST",parms) || {"data":[]};
		var newData = [];
    	for(var i=0; i<data.data.length; i++){
    		var jsonData = data.data[i];
    		for(var item in jsonData){
        		newData.push({
        			"id" : item,
        			"value" : jsonData[item][0],
        			"name" : jsonData[item][2],
        			"class" : jsonData[item][1],
        			"type" : "select",
					"data":[
						{
							"name":jsonData[item][2],
							"value":jsonData[item][0]
						}
					]
        		})
    		}
    	}
    	createTask.stepForm.stepForm2 = newData;
    	var testModel = `
			<!--遍历循环步骤二信息-->
            ${newData.map(f => `
				<div class="form-group">
					<label for="${f.id}" class="col-sm-6 control-label">${f.name} :</label>
					<div class="col-sm-6">
						<select class="form-control" id="${f.id}">
				            ${createMap.formValue.map(f => `
								<option value="${f.value}">${f.name}</option>
				            `).join('')}
			            </select>
					</div>
				</div>
            `).join('')}
        `;
        $(".stepForm2").html(testModel);
        
        for(var i=0; i<newData.length; i++){
        	var id = newData[i].id;
        	var value = newData[i].value;
        	var domValue = $("#"+id+" option");
        	for(var l=0; l<domValue.length; l++){
	        	if(domValue[l].value == value){
					domValue[l].selected=true;
					break;
	        	}
        	}
        }
        
	},
	
}
