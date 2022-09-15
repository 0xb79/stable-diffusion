import requests
from requests.models import Response

class Sdrequests:
    def __init__(self, secret="stablediffusion", verify_ssl=False):
        self.secret = secret
        self.verify_ssl = verify_ssl
        self.wrong_secret = requests.models.Response()
        self.wrong_secret.status_code = 500
        self.wrong_secret._content = b'secret does not match'
        
    def get(url="", params=None, data=None, headers={}):
        headers["Credentials"] = self.secret
        resp = requests.get(t_uri+url, params=params, data=data, headers=headers, verify=verify_ssl)
        
        if match_response_secret(resp):
            return resp
        else:
            return self.wrong_secret
        
        return resp
        
    def post(url="", params=None, data=None, headers={}):
        headers["Credentials"] = self.secret
        resp = requests.post(t_uri+url, params=params, data=data, headers=headers, verify=verify_ssl)
        
        if match_response_secret(resp):
            return resp
        else:
            return self.wrong_secret
            
    def match_request_secret(req):
        if req.headers.get("Credentials") == self.secret:
            return True
        else:
            return False
    
    def match_response_secret(resp):
        if resp.headers["Credentials"] == self.secret:
            return True
        else:
            return False