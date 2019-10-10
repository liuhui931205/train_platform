
//--------------------------------------用户登录(  判断跳转   )---------------------------------
var user = {
	ouc: {
		user: null,
		oauth2User: window.Kd_auth_server+"oauth",
		clientId: "client",
		clientSecret: "secret"
	},
	userId: null,				//登录用户的id
	username: null,				//登录用户名
	showname: null,				//登录用户的显示名称
	authority: null,			//登录的用户类型
	userImg: null,				//登录的用户头像
	officeCode: null,			//登录用户所对应的code编号
	
//	登录系统的判断
	userData: function(){
//		获取所登录的用户
		$.ajax( {
		    type : "get",
		    url : window.Kd_auth_server+"user",
		    async : false,
		    data : {},
		    success : function(data){
//		    	获取不到登录用户则进行跳转登录
		    	if(!data){
		    		user.oAuthLoginPage()
		    		return ;
		    	}
		    	
//		  		获取用户iD
		    	if(data.principal.userId){
		    		user.userId = data.principal.userId;
		    	}
//		  		获取用户名
		    	if(data.principal.username){
		    		user.username = data.principal.username;
		    	}
//		  		获取用户昵称
		    	if(data.principal.showname){
		    		user.showname = data.principal.showname;
		    	}
//		  		获取用户code参数
		    	if(data.principal.officeCode){
		    		user.officeCode = data.principal.officeCode;
		    	}
//		  		获取用户头像图片
		    	if(data.principal.showimg){
		    		user.userImg = data.principal.showimg;
		    	}else{
		    		user.userImg = "dist/img/avatar.png";
		    	}
//				通过获取的昵称来进行修改页面上显示名称
				if(data.principal.username){
//					$(".hidden-xs").html(data.principal.showname);
					$(".userName").html(data.principal.showname);
				}
//				通过获取的头像图片来进行修改页面上显示图像
				if(user.userImg){
//					$(".user-image")[0].src = user.userImg;
					for(var i=0; i<$(".img-circle").length; i++){
						$(".img-circle")[i].src = user.userImg;
					}
				}
		    },
		   	error: function(XMLHttpRequest, textStatus, errorThrown) {
		   		console.log(XMLHttpRequest.status+"---"+XMLHttpRequest.readyState+"---"+textStatus);
		   		user.oAuthLoginPage();
		   		return ;
		   	},
		});
	},
	
	getNoSearchUrl: function(){
		return window.location.origin + window.location.pathname;
	},
	
	getNoHashUrl: function(){
		return window.location.href.split("#")[0];
	},
	
	isNotNull: function(str){
		return str != null && $.trim(str) != "";
	},
	
	oAuthLoginPage: function(){
		$.oauth2Client.begin({
			authorizeUrl: user.ouc.oauth2User + "/authorize",
			clientId: user.ouc.clientId,
			redirectUri: user.getNoHashUrl()
		});
	},
	
	signout: function(){
		$.ajax({
			type:"get",
			url:window.Kd_auth_server+"logout",
			async:false,
			success : function(data) {
//				登出成功，刷新页面(会自动跳转至登陆界面)
				window.location.reload();
		    },
		   	error: function(XMLHttpRequest, textStatus, errorThrown) {
		   		console.log(XMLHttpRequest.status+"---"+XMLHttpRequest.readyState+"---"+textStatus)
		   	},
		});
	},
}
user.userData();
