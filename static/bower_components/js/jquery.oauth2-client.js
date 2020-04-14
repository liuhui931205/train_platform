
// 检测浏览器是否支持跨域的变量
// jQuery.support.cors = true;
(function($) {
	$.extend({
		oauth2Client : {
			begin : function(opts) {
				opts = $.extend({
					clientId : false,
					authorizeUrl : false,
					redirectUri : window.location.href,
					responseType : 'code'
				}, opts);
				window.location.href = opts.authorizeUrl + '?client_id='
						+ opts.clientId + '&response_type=' + opts.responseType
						+ '&redirect_uri=' + opts.redirectUri;
			},
			end : function(opts, fn, errFn) {
				opts = $.extend({
					code : false,
					clientId : false,
					clientSecret : false,
					accessTokenUrl : false,
					redirectUri : window.location.href,
					userInfoUrl : false
				}, opts);
				if (!opts.code)
					return;
				
				$.ajax({
					async:false,
					url: opts.accessTokenUrl,
					type: "post",
//					dataType:'jsonp',
//					jsonpCallback:'callback',
					data: {
						grant_type : "authorization_code",
						client_id : opts.clientId,
						client_secret : opts.clientSecret,
						code : opts.code,
						redirect_uri : opts.redirectUri
					},
					success: function(data) {
						if (data) {
							if (typeof data === 'string') {
								data = $.parseJSON(data);
							}
							var accessToken = data.access_token;
							var expiresIn = data.expires_in;
						}
						fn(accessToken, expiresIn);
					},
					error: function(XMLHttpRequest, textStatus, errorThrown){
						errFn(XMLHttpRequest, textStatus, errorThrown);
					},
					complete: function(XMLHttpRequest, textStatus){
					}
				});
				
			}
		}
	});
	/*
	function getNormalJson(data,left,right){
		var seqIndex = data.indexOf("\{");
		var seq2Index = data.indexOf("\[");
		// JSON obj格式
		if(seqIndex!=-1 && seqIndex < seq2Index){
			data = data.substring(data.indexOf("\{"), data.lastIndexOf("\}") + 1);
		}
		// JSON array格式
		else if(seq2Index!=-1 && seq2Index < seqIndex){
			data = data.substring(data.indexOf("\["), data.lastIndexOf("\]") + 1);
		}
		if(left && right){
			data = data.substring(data.indexOf(left)+1, data.lastIndexOf(right) );
		}
		return data;
	}
	*/
	
})(jQuery);
