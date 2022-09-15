from flask import Flask, request, send_file, Response, make_response
from torch import autocast, cuda
from urlparse import urlparse
import torch, os, requests, io, uuid
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
import sdhttp
from setup_logger import logger

app = Flask(__name__)
sdrequests = sdhttp.Sdrequests()

workers = {}

@app.before_request 
def check_secret():
    if sdrequests.match_request_secret(request) == False:
        return make_response("secret does not match",500)

@app.route("/sd/txt2img")
def process_txt2img():
    prompt = request.values.get("prompt")
    if (prompt == None or prompt == ""):
        return make_response("Need to input prompt", 400)
    
    guidance = request.values.get("guidance")
    if guidance != None:
        if guidance.isnumeric() == False:
            return make_response("Need to input numeric guidance setting")
    else:
        guidance = 7.5
    
    iterations = request.values.get("iterations")
    if iterations != None:
        if iterations.isnumeric() == False:
            return make_response("Need to input numeric iterations setting")
    else:
        iterations = 50
    
    height = request.values.get("height")
    if height != None:
        if height.isnumeric() == False:
            return make_response("Need to input numeric height")
        if (height % 64) != 0:
            return make_response("Need to input height increments of 64")
    else:
        height = 512
    
    width = request.values.get("width")
    if width != None:
        if width.isnumeric() == False:
            return make_response("Need to input numeric width")
        if (width % 64) != 0:
            return make_response("Need to input width increments of 64")
    else:
        width = 512
    
    seed = request.values.get("seed")
    if seed == None:
        seed = ''
    
    req = request.full_path
    resp = sdrequests.get("/txt2img",headers={"prompt_id":uuid.uuid4()},params={"prompt":prompt,"guidance":guidance,"iterations":iterations,"height":height,"width":width,"seed":seed})
    
    if resp.status_code == 200:
        app.logger.info("image received from worker")
        return send_file(resp.content, mimetype='image/png')
    elif resp.status_code == 503:
        app.logger.info("worker busy")
        return make_response(resp.content, resp.status_code)
    else:
        app.logger.info("error from worker")
        return make_response("could not process prompt", 500)

@app.route("/registerworker", methods=['POST'])
def register_worker():
    w_config = request.get_json()
    workers[w_config["id"]] = w_config}

@app.route("/setworkerconfig")
def set_worker_config():
    w_url = request.values.get("url")
    token = request.values.get("token")
    sessions = request.values.get("sessions")
    gpu = request.values.get("gpu")
    mh = request.values.get("maxheight")
    mw = request.values.get("maxwidth")
    w_id = request.values.get("id")
    
    if w_uri != None:
        if "https://" in w_uri:
            try:
                result = urlparse(x)
                sdrequests.w_uri = w_uri
            except:
                return make_response("w_uri provided is not a url", 400)
        else:
            return make_response("w_uri must include https://", 400)
    
    if token != None:
        resp = sdrequests.post("/accesstoken", data={'token':token})
        if resp.status_code != 200:
                return make_response(resp.content, resp.status_code)
    
    if sessions != None:
        resp = sdrequests.post("/maxsessions", data={'sessions':sessions})
        if resp.status_code != 200:
            return make_response(resp.content, resp.status_code)
    
    if gpu != None:
        resp = sdrequests.post("/gpu", data={'gpu':gpu})
        if resp.status_code != 200:
            return make_response(resp.content, resp.status_code)
    
    if mh != None or mw != None:
        h = 512
        w = 512
        if mh != None:
            h = mh
        if mw != None:
            w = mw
        resp = sdrequests.post("/maximagesize", data={"maxheight":h,"maxwidth":w})
        
    return make_response("worker config updated",200)

#def root_dir():
#    return os.path.abspath(os.path.dirname(__file__))

def main(argv):
    secret = "stablediffusion"
    ip = "127.0.0.1"
    p = "5555"
    
    try:
      opts, args = getopt.getopt(argv,"",["secret","ip","port","noselfsignedcert"])
    except getopt.GetoptError:
        app.logger.info("error reading options")
    for opt, arg in opts:
        if opt == '--secret':
            app.logger.info("secret set")
            sdrequests.secret = arg
        elif opt == "--ip":
            app.logger.info("ip set")
            ip = arg
        elif opt == "--port":
            app.logger.info("port set")
            p = arg
        elif: opt == "--noselfsignedcert"
            sdrequests.verify_ssl = True
        
    app.run(host=ip, port=p)
     
if __name__=="__main__":
    main(sys.argv[1:])