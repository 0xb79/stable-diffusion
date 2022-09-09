import requests

class Sdrequests:
    def __init__(self, t_uri="https://127.0.0.1:5555", secret="stablediffusion", verify_ssl=False):
        self.t_uri = t_uri
        self.secret = secret
        self.verify_ssl = verify_ssl
        
    def get(self, url="", params=None, data=None, headers={}):
        headers["Credentials"] = self.secret
        resp = requests.get(t_uri+url, params=params, data=data, headers=headers, verify=verify_ssl)
        return resp
        
    def post(self, url="", params=None, data=None, headers={}):
        headers["Credentials"] = self.secret
        resp = requests.post(t_uri+url, params=params, data=data, headers=headers, verify=verify_ssl)
        return resp