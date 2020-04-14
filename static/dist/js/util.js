
/*
 * 
 * 经常用到的事件写为通用方法
 * 
 */

var util = {

//	时间格式方法(包括时分秒)
	Time: function(value,judge=false){
		var dateTime = new Date(value);
	    var y = dateTime.getFullYear();			//年
	    var m = dateTime.getMonth() + 1;		//月
	    m = m < 10 ? '0' + m : m;
	    var d = dateTime.getDate();				//日
	    d = d < 10 ? ('0' + d) : d;
	    var h = dateTime.getHours();			//时
	    h = h < 10 ? ('0' + h) : h;
	    var t = dateTime.getMinutes();			//分
	    t = t < 10 ? ('0' + t) : t;
	    var s = dateTime.getSeconds();			//秒
	    s = s < 10 ? ('0' + s) : s;
	    if(judge){
			return y + '-' + m + '-' + d + ' ' + h + ':' + t;
	    }else{
			return y + '-' + m + '-' + d + ' ' + h + ':' + t + ':' + s;
	    }
	},
	
//	时间格式方法(包括毫秒数)
	Millisecond: function(value){
		var dateTime = new Date(value);
	    var y = dateTime.getFullYear();			//年
	    var m = dateTime.getMonth() + 1;		//月
	    m = m < 10 ? '0' + m : m;
	    var d = dateTime.getDate();				//日
	    d = d < 10 ? ('0' + d) : d;
	    var h = dateTime.getHours();			//时
	    h = h < 10 ? ('0' + h) : h;
	    var t = dateTime.getMinutes();			//分
	    t = t < 10 ? ('0' + t) : t;
	    var s = dateTime.getSeconds();			//秒
	    s = s < 10 ? ('0' + s) : s;
		return y + '-' + m + '-' + d + ' ' + h + ':' + t + ':' + s + '.' + dateTime.getMilliseconds();
	},
	
//	时间格式方法(只包括年月日)
	TimeData: function(date){
	    var y = date.getFullYear();  
	    var m = date.getMonth() + 1;  
	    m = m < 10 ? '0' + m : m;  
	    var d = date.getDate();  
	    d = d < 10 ? ('0' + d) : d;  
	    return y + '-' + m + '-' + d;
	},

//	获取列表的最新数据--当前任务与我的任务(数据未获取到时,按钮置灰不可用)
	Refresh: function(value){
		$("#refresh button").attr("disabled", true);
		$("#refresh button").html("刷新列表...");
		setTimeout(function(){
			value.Table();
		},100)
	},
	
	errorView: function(view,junge=false){
		var errorModel = `
			<span class="errorUpload" title="提示信息:${view}!">提示信息:${view}!</span>
		`;
		if(junge){
			errorModel = `
				<span class="successpload" title="提示信息:${view}!">提示信息:${view}!</span>
			`;
		}
		$(".upload").html(errorModel);
		$(".upload").fadeIn(300);
		setTimeout(function(){
			$(".upload").fadeOut(300);
		},2000);
	},
	
//	转换json格式字符串
	formatJson:function(json) {
		var formatted = JSON.stringify(JSON.parse(json), null, 4);
        return formatted;
    },
    
//  获取ajax数据
    newAjax: function(url,type,data,callback=false){
    	$("#loading").css("display","block");
    	var jsonData;
    	var judge = false;
    	if(callback){
    		judge = true;
    	}
    	if(type=="POST"){
//			提交创建任务数据
		    $.ajax( {
		        type : "POST",
		        url : url,
	        	async : judge,
				contentType: "application/json; charset=utf-8",
		        data : JSON.stringify(data),
		        success : function(msg) {
		        	jsonData = msg;
		        	if(callback){
    					$("#loading").css("display","none");
		        		callback(jsonData);
		        	}
		        },
		        error: function(XMLHttpRequest, textStatus, errorThrown){
		            util.errorView("系统异常");
    				$("#loading").css("display","none");
			   		console.log(XMLHttpRequest.status+"---"+XMLHttpRequest.readyState);
			   		return;
		        }
		    });
    	}else{
	    	$.ajax( {
	        	type : "GET",
		        url : url,
		        async : judge,
		        data : {},
		        success : function(msg) {
		        	jsonData = msg;
		        	if(callback){
    					$("#loading").css("display","none");
		        		callback(jsonData);
		        	}
		        },
			   	error: function(XMLHttpRequest, textStatus, errorThrown) {
		            util.errorView("系统异常");
    				$("#loading").css("display","none");
			   		console.log(XMLHttpRequest.status+"---"+XMLHttpRequest.readyState);
			   		return;
			   	},
		    });
    	}
    	if(!callback){
    		$("#loading").css("display","none");
    		return jsonData;
    	}
    },
    
    initKnob: function(id){
		$("."+id).knob({
	      draw: function () {
	        if (this.$.data('skin') == 'tron') {
	
	          var a = this.angle(this.cv)  // Angle
	              , sa = this.startAngle          // Previous start angle
	              , sat = this.startAngle         // Start angle
	              , ea                            // Previous end angle
	              , eat = sat + a                 // End angle
	              , r = true;
	
	          this.g.lineWidth = this.lineWidth;
	
	          this.o.cursor
	          && (sat = eat - 0.3)
	          && (eat = eat + 0.3);
	
	          if (this.o.displayPrevious) {
	            ea = this.startAngle + this.angle(this.value);
	            this.o.cursor
	            && (sa = ea - 0.3)
	            && (ea = ea + 0.3);
	            this.g.beginPath();
	            this.g.strokeStyle = this.previousColor;
	            this.g.arc(this.xy, this.xy, this.radius - this.lineWidth, sa, ea, false);
	            this.g.stroke();
	          }
	
	          this.g.beginPath();
	          this.g.strokeStyle = r ? this.o.fgColor : this.fgColor;
	          this.g.arc(this.xy, this.xy, this.radius - this.lineWidth, sat, eat, false);
	          this.g.stroke();
	
	          this.g.lineWidth = 2;
	          this.g.beginPath();
	          this.g.strokeStyle = this.o.fgColor;
	          this.g.arc(this.xy, this.xy, this.radius - this.lineWidth + 1 + this.lineWidth * 2 / 3, 0, 2 * Math.PI, false);
	          this.g.stroke();
	
	          return false;
	        }
	      }
	    });
    }
    
    
//	range: function(value){
//		var replaces = value.replace("POLYGON((","").replace("))","");
//		return replaces.split(",");
//	},
	
}
