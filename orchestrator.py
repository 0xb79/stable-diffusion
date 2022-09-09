from flask import Flask, request, send_file, Response, make_response
from torch import autocast, cuda
from urlparse import urlparse
import torch, os, requests, io
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
from .sdhttp import Sdrequests

app = Flask(__name__)

sdrequests = Sdrequests()

@app.route("/stable_diffusion")
def get_result():
    prompt = request.values.get("prompt")
    if (prompt == None or prompt == ""):
        return make_response("Need to input prompt", 400)
        
    req = request.full_path
    resp = sdrequests.get(req)
    
    if resp.status_code == 200 :
        print("image received from transcoder")
        return send_file(io.BytesIO(resp.content), mimetype='image/png')
    elif resp.status_code == 503:
        print("transcoder busy")
        return make_response(resp.content, resp.status_code)
    else:
        print("error from transcoder")
        return make_response("could not process prompt", 500)

@app.route("/settranscoderconfig")
def set_transcoder_config():
    t_uri = request.values.get("t_uri")
    token = request.values.get("token")
    sessions = request.values.get("sessions")
    gpu = request.values.get("gpu")
    
    if t_uri != None:
        if "https://" in t_uri:
            try:
                result = urlparse(x)
                sdrequests.t_uri = t_uri
            except:
                return make_response("t_uri provided is not a url", 400)
        else:
            return make_response("t_uri must include https://", 400)
    
    if token != None:
        resp = sdrequests.post("/setaccesstoken", data={'token':token})
        if resp.status_code != 200:
            return make_response(resp.content, resp.status_code)
    
    if sessions != None:
        resp = sdrequests.post("/setmaxsessions", data={'sessions':sessions})
        if resp.status_code != 200:
            return make_response(resp.content, resp.status_code)
    
    if gpu != None:
        resp = sdrequests.post("/setgpu", data={'gpu':gpu})
        if resp.status_code != 200:
            return make_response(resp.content, resp.status_code)
    
    return make_response("transcoder config updated",200)   
    
def root_dir():
    return os.path.abspath(os.path.dirname(__file__))

def main(argv):
    secret = "stablediffusion"
    ip = "127.0.0.1"
    p = "5555"
    
    try:
      opts, args = getopt.getopt(argv,"",["secret","ip","port","noselfsignedcert"])
    except getopt.GetoptError:
        print("error reading options")
    for opt, arg in opts:
        if opt == '--secret':
            print("secret set")
            sdrequests.secret = arg
        elif opt == "--ip":
            print("ip set")
            ip = arg
        elif opt == "--port":
            print("port set")
            p = arg
        elif: opt == "--noselfsignedcert"
            sdrequests.verify_ssl = True
        
    app.run(host=ip, port=p)
     
if __name__=="__main__":
    main(sys.argv[1:])