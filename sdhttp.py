import requests
from requests.models import Response
from requests.packages.urllib3.exceptions import InsecureRequestWarning
from flask import make_response, send_file

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

class InternalRequests:
    def __init__(self, secret="stablediffusion", verify_ssl=False):
        self.secret = secret
        self.verify_ssl = verify_ssl
        self.wrong_secret = requests.models.Response()
        self.wrong_secret.status_code = 400
        self.wrong_secret._content = b'secret does not match'
        self.http_error = requests.models.Response()
        self.http_error.status_code = 404
        self.http_error._content = b'error'
        
    def get(self, url="", params=None, data=None, headers={}):
        headers["Credentials"] = self.secret
        try:
            resp = requests.get(url, params=params, data=data, headers=headers, verify=self.verify_ssl, timeout=90)
            if self.match_response_secret(resp):
                return resp
            else:
                return self.wrong_secret
        except requests.exceptions.HTTPError as errh:
            return self.http_error
        except requests.exceptions.ConnectionError as errc:
            return self.http_error
        except requests.exceptions.Timeout as errt:
            return self.http_error
        except requests.exceptions.RequestException as err:
            return self.http_error
        
    def post(self, url="", params=None, data=None, headers={}, json={}):
        headers["Credentials"] = self.secret
        try:
            resp = requests.post(url, params=params, data=data, headers=headers, json=json, verify=self.verify_ssl, timeout=90)
            if self.match_response_secret(resp):
                return resp
            else:
                return self.wrong_secret
        except requests.exceptions.HTTPError as errh:
            return self.http_error
        except requests.exceptions.ConnectionError as errc:
            return self.http_error
        except requests.exceptions.Timeout as errt:
            return self.http_error
        except requests.exceptions.RequestException as err:
            return self.http_error
        
    def send_file_with_secret(self, data, type="img/png", filename="", resp=None, addl_headers=None):
        if resp == None:
            send_file_w_secret = send_file(data, mimetype=type, download_name=filename)
            send_file_w_secret.headers["Credentials"] = self.secret
            if addl_headers != None:
                send_file_w_secret.headers = {**send_file_w_secret.headers, **addl_headers}
            return send_file_w_secret
        else:
            resp.headers["Credentials"] = self.secret
            if addl_headers != None:
                resp.headers = {**resp.headers, **addl_headers}
            return resp
        
    def make_response_with_secret(self, msg, status, resp=None, addl_headers=None):
        if resp == None:
            resp_w_secret = make_response(msg, status)
            resp_w_secret.headers["Credentials"] = self.secret
            if addl_headers != None:
                resp_w_secret.headers = {**resp_w_secret.headers, **addl_headers}
            return resp_w_secret
        else:
            resp.headers["Credentials"] = self.secret
            if addl_headers != None:
                resp.headers = {**resp.headers, **addl_headers}
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