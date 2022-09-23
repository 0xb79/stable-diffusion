import requests
from requests.models import Response
from requests.packages.urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

class Sdrequests:
    def __init__(self, secret="stablediffusion", verify_ssl=False):
        self.secret = secret
        self.verify_ssl = verify_ssl
        self.wrong_secret = requests.models.Response()
        self.wrong_secret.status_code = 500
        self.wrong_secret._content = b'secret does not match'
        
    def get(self, url="", params=None, data=None, headers={}):
        headers["Credentials"] = self.secret
        resp = requests.get(url, params=params, data=data, headers=headers, verify=self.verify_ssl)
        
        if self.match_response_secret(resp):
            return resp
        else:
            return self.wrong_secret
        
        return resp
        
    def post(self, url="", params=None, data=None, headers={}, json={}):
        headers["Credentials"] = self.secret
        resp = requests.post(url, params=params, data=data, headers=headers, json=json, verify=self.verify_ssl)
        if self.match_response_secret(resp):
            return resp
        else:
            return self.wrong_secret
    
    def send_file_with_secret(self, resp):
        resp.headers["Credentials"] = self.secret
        return resp
        
    def make_response_with_secret(self, resp):
        resp.headers["Credentials"] = self.secret
        return resp
    
    def match_request_secret(self, req):
        if "Credentials" in req.headers and req.headers["Credentials"] == self.secret:
            return True
        else:
            return False
    
    def match_response_secret(self, resp):
        if "Credentials" in resp.headers and resp.headers.get("Credentials") == self.secret:
            return True
        else:
            return False