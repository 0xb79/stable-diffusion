from flask import Flask, request, send_file, Response, make_response
from torch import autocast, cuda
import sys, os, io, logging, uuid, torch, requests, argparse, json
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
import sdhttp

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

config = {"sort_workers_by":"load"}

sdrequests = sdhttp.Sdrequests()

workers = {}

#@app.before_request 
#def check_secret():
#    if sdrequests.match_request_secret(request) == False:
#        return make_response("secret does not match", 500)

@app.route("/txt2img", methods=['GET'])
def process_txt2img():
    prompt = request.values.get("prompt")
    app.logger.info("processing txt2img prompt: " + prompt)
    if (prompt == None or prompt == ""):
        app.logger.info("prompt not set")
        return make_response("Need to input prompt", 400)
    
    guidance = request.values.get("guidance")
    if guidance != None:
        if guidance.isnumeric() == False:
            return make_response("Need to input numeric guidance setting", 400)
    else:
        guidance = 7.5
    
    iterations = request.values.get("iterations")
    if iterations != None:
        if iterations.isnumeric() == False:
            return make_response("Need to input numeric iterations setting", 400)
    else:
        iterations = 50
    
    height = request.values.get("height")
    if height != None:
        if height.isnumeric() == False:
            return make_response("Need to input numeric height", 400)
        if (height % 64) != 0:
            return make_response("Need to input height increments of 64", 400)
    else:
        height = 512
    
    width = request.values.get("width")
    if width != None:
        if width.isnumeric() == False:
            return make_response("Need to input numeric width", 400)
        if (width % 64) != 0:
            return make_response("Need to input width increments of 64", 400)
    else:
        width = 512
    
    batch_size = request.values.get("batch_size")
    if batch_size != None:
        if batch_size.isnumeric() == False:
            return make_response("need to input numeric batch_size", 400)
    else:
        batch_size = 3
    
    seed = request.values.get("seed")
    if seed == None:
        seed = ''
    
    seed_step = request.values.get("seed_step")
    if seed_step != None:
        if seed_step.isnumeric() == False:
            return make_response("need to input numeric seed_step", 400)
        else:
            seed_step = 1
    else:
        seed_step = 1
    
    #select worker to send to
    if len(workers) == 0:
        return make_response("no workers registered", 500)
        
    req = request.full_path
    wkr = select_worker('load')
    print(str(wkr))
    resp = sdrequests.get(wkr['url']+"/txt2img",headers={"prompt_id":str(uuid.uuid4())},params={"prompt":prompt,"batch_size":batch_size,"guidance":guidance,"iterations":iterations,"height":height,"width":width,"seed":seed, "seed_step":seed_step})
    
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
    app.logger.info("worker registration received: "+str(w_config))
    if w_config["url"] == "":
        resp = sdrequests.make_response_with_secret('url must be set',400)
        return (resp.text, resp.status_code, resp.headers.items())
    else:
        workers[w_config["id"]] = {'config':w_config,'load':0,'score':0, 'resp_time':[], 'error_cnt':0}
        app.logger.info("worker registered  (id: "+w_config["id"]+")")
        resp = sdrequests.make_response_with_secret('worker registered',200)
        return (resp.text, resp.status_code, resp.headers.items())
    
def select_worker(sort_by):
    sort_workers = []
    if sort_by == 'load':
        sort_workers = sorted(workers.items(), key=lambda x: x[1]['load'], reverse=False)
    elif sort_by == 'score':
        sort_workers = sorted(workers.items(), key=lambda x: x[1]['score'], reverse=True)
    elif sort_by == 'error_cnt':
        sort_workers = sorted(workers.items(), key=lambda x: x[1]['error_cnt'], reverse=False)
    
    return sort_workers[0][1]["config"]

def remove_worker(id):
    try:
        del workers[id]
        app.logger.info("worker "+id+" removed")
    except:
        app.logger.info("could not remove worker "+id+". worker id not found")

@app.route("/setworkerconfig")
def set_worker_config():
    token = request.values.get("token")
    sessions = request.values.get("sessions")
    gpu = request.values.get("gpu")
    mh = request.values.get("maxheight")
    mw = request.values.get("maxwidth")
    w_id = request.values.get("id")
    w_url = workers[w_id]["url"]
    
    if token != None:
        resp = sdrequests.post(w_url+"/accesstoken", data={'token':token})
        if resp.status_code != 200:
                return make_response(resp.content, resp.status_code)
    
    if sessions != None:
        resp = sdrequests.post(w_url+"/maxsessions", data={'sessions':sessions})
        if resp.status_code != 200:
            return make_response(resp.content, resp.status_code)
    
    if gpu != None:
        resp = sdrequests.post(w_url+"/gpu", data={'gpu':gpu})
        if resp.status_code != 200:
            return make_response(resp.content, resp.status_code)
    
    if mh != None or mw != None:
        h = 512
        w = 512
        if mh != None:
            h = mh
        if mw != None:
            w = mw
        resp = sdrequests.post(w_url+"/maximagesize", data={"maxheight":h,"maxwidth":w})
        
    return make_response("worker config updated",200)

#def root_dir():
#    return os.path.abspath(os.path.dirname(__file__))

def main(args):
    sdrequests.secret = args.secret
    sdrequests.verify_ssl = args.noselfsignedcert
    
    app.logger.info("orchestrator config set, starting node")
    app.run(host=args.ipaddr, port=args.port)
     
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ipaddr", type=str, action="store", default="127.0.0.1")
    parser.add_argument("--port", type=str, action="store", default="5555")
    parser.add_argument("--secret", type=str, action="store", default="stablediffusion")
    parser.add_argument("--noselfsignedcert", type=bool, action="store", default=False)
    args = parser.parse_args()
    
    main(args)