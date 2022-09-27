from flask import Flask, request, send_file, Response, make_response, jsonify
from torch import autocast, cuda
from functools import wraps
import sys, os, io, logging, uuid, torch, requests, argparse, json, time, copy, threading
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
import sdhttp
from helpers import is_number

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

config = {"sort_workers_by":"load","managesecret":""}
sdr = sdhttp.InternalRequests()
workers = {}

def internal(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if "Credentials" in request.headers and sdr.match_request_secret(request):
            return f(*args, **kwargs)
        else:
            return make_response("secret does not match", 400)
    return wrap
    
def manager(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if "Credentials" in request.headers and request.headers["Credentials"] == config["managesecret"]:
            return f(*args, **kwargs)
        else:
            return make_response("secret does not match", 400)
    return wrap
    
@app.route("/txt2img", methods=['GET','POST'])
def process_txt2img():
    prompt = request.values.get("prompt")
    app.logger.info("processing txt2img prompt: " + prompt)
    if (prompt == None or prompt == ""):
        app.logger.info("prompt not set")
        return make_response("Need to input prompt", 400)
    
    guidance = request.values.get("guidance")
    if guidance != None:
        if is_number(guidance) == False:
            return make_response("Need to input numeric guidance setting", 400)
    else:
        guidance = 7.5
    
    iterations = request.values.get("iterations")
    if iterations != None:
        if is_number(iterations) == False:
            return make_response("Need to input numeric iterations setting", 400)
    else:
        iterations = 50
    
    height = request.values.get("height")
    if height != None:
        if is_number(height) == False:
            return make_response("Need to input numeric height", 400)
        if (height % 64) != 0:
            return make_response("Need to input height increments of 64", 400)
    else:
        height = 512
    
    width = request.values.get("width")
    if width != None:
        if is_number(width) == False:
            return make_response("Need to input numeric width", 400)
        if (width % 64) != 0:
            return make_response("Need to input width increments of 64", 400)
    else:
        width = 512
    
    batch_size = request.values.get("batchsize")
    if batch_size != None:
        if is_number(batch_size) == False:
            return make_response("need to input numeric batchsize", 400)
    else:
        batch_size = 1
    
    seed = request.values.get("seed")
    if seed == None:
        seed = ''
    
    seed_step = request.values.get("seedstep")
    if seed_step != None:
        if is_number(seed_step) == False:
            return make_response("need to input numeric seedstep", 400)
        else:
            seed_step = 1
    else:
        seed_step = 1
    
    #select worker to send to
    if len(workers) == 0:
        return make_response("no workers registered", 500)
        
    req = request.full_path
    prompt_id = str(uuid.uuid4())
    worker = select_worker('load')
    if worker != None:
        app.logger.info("worker selected: "+worker["id"])
        start = time.time()
        try:
            resp = sdr.post(worker['url']+"/txt2img",headers={"prompt_id":prompt_id},params={"prompt":prompt,"batchsize":batch_size,"guidance":guidance,"iterations":iterations,"height":height,"width":width,"seed":seed, "seedstep":seed_step})
        
            if resp.status_code == 200:
                app.logger.info("image received from worker "+str(worker["id"]))
                took = int(time.time() - start)
                worker_done(worker["id"], resp_time=took)
                
                img = io.BytesIO(resp.content)
                return send_file(img, mimetype='image/png',download_name=prompt_id+".png")
            elif resp.status_code == 503:
                app.logger.info("worker busy")
                worker_done(worker["id"])
                
                return make_response(resp.content, resp.status_code)
            else:
                app.logger.info("error from worker: "+resp.text)
                worker_done(worker["id"], True)
                return make_response("could not process prompt", 500)
        except Exception as ee:
            worker_done(worker["id"], True)
            return make_response("could not process prompt", 500)
    else:
        app.logger.info("no worker available")
        return make_response("no workers available",503)

@app.route("/img2img", methods=['POST'])
def process_img2img():
    prompt = request.values.get("prompt")
    app.logger.info("processing img2img prompt: " + prompt)
    if (prompt == None or prompt == ""):
        app.logger.info("prompt not set")
        return make_response("Need to input prompt", 400)
    
    img = io.BytesIO(request.files.get("init_img"))
    if img == None:
        return make_response("no init_img provided", 400)
        
    guidance = request.values.get("guidance")
    if guidance != None:
        if is_number(guidance) == False:
            return make_response("Need to input numeric guidance setting", 400)
    else:
        guidance = 7.5
    
    strength = request.values.get("strength")
    if strength != None:
        if is_number(strength) == False:
            return make_response("Need to input numeric strength setting", 400)
    else:
        strength = .75
    
    iterations = request.values.get("iterations")
    if iterations != None:
        if is_number(iterations) == False:
            return make_response("Need to input numeric iterations setting", 400)
    else:
        iterations = 50
    
    seed = request.values.get("seed")
    if seed == None:
        seed = ''
    
    batch_size = request.values.get("batchsize")
    if batch_size != None:
        if is_number(batch_size) == False:
            return make_response("need to input numeric batchsize", 400)
    else:
        batch_size = 1
    
    #select worker to send to
    if len(workers) == 0:
        return make_response("no workers registered", 500)
        
    req = request.full_path
    prompt_id = str(uuid.uuid4())
    worker = select_worker('load')
    if worker != None:
        app.logger.info("worker selected: "+worker["id"])
        start = time.time()
        try:
            resp = sdr.get(worker['url']+"/img2img",headers={"prompt_id":prompt_id},params={"prompt":prompt,"batchsize":batch_size,"guidance":guidance,"strength":strength,"iterations":iterations, "seed":seed})
        
            if resp.status_code == 200:
                app.logger.info("image received from worker")
                took = int(time.time() - start)
                worker_done(worker["id"], resp_time=took)
                
                img = io.BytesIO(resp.content)
                return send_file(img, mimetype='image/png',download_name=prompt_id+".png")
            elif resp.status_code == 503:
                app.logger.info("worker busy")
                worker_done(worker["id"])
                
                return make_response(resp.content, resp.status_code)
            else:
                app.logger.info("error from worker")
                worker_done(worker["id"], True)
                return make_response("could not process prompt", 500)
        except Exception as ee:
            worker_done(worker["id"], True)
            return make_response("could not process prompt", 500)
            
    else:
        app.logger.info("no worker available")
        return make_response("no workers available",503)
        
@app.route("/registerworker", methods=['POST'])
@internal
def register_worker():
    global workers
    w_config = request.get_json()
    w_ip = worker_ip(request)
    app.logger.info("worker registration received: "+str(w_config))
    if w_config["url"] == "":
        resp = sdr.make_response_with_secret('url must be set',404)
        return resp
    else:
        if w_config["id"] in workers.keys():
            if workers[w_config["id"]]["remote_addr"] != w_ip:
                app.logger.info("worker id ("+w_config["id"]+") already registered at different ip address")
                return sdr.make_response_with_secret('id already in use',400)
        
        workers[w_config["id"]] = {'config':w_config,'load':0,'score':[], 'resp_time':[], 'error_cnt':0, "remote_addr":w_ip, "last_checkin":0,"last_status_check":0}
        app.logger.info("worker registered  (id: "+w_config["id"]+")")
        resp = sdr.make_response_with_secret('worker registered',200)
        return resp

@app.route("/workerisregistered/<id>", methods=['GET'])
@internal
def worker_is_registered(id):
    global workers
    w_ip = worker_ip(request)
    if id in workers.keys():
        if workers[id]["remote_addr"] == w_ip:
            workers[id]["last_checkin"] = time.time()
            return sdr.make_response_with_secret("",200)
        else:
            return sdr.make_response_with_secret("worker id already registered a differnet ip address",404)
    else:
        app.logger.info("worker "+str(id)+" not registered, expecting registration request")
        return sdr.make_response_with_secret("",400)

def monitor_workers():
    global workers
    while True:
        time.sleep(10)
        try:
            del_workers=[]
            for w in workers.keys():
                resp = sdr.get(workers[w]["config"]["url"]+"/workerstatus")
                if resp.status_code == 200:
                    if is_number(resp.text):
                        if workers[w]["load"] != int(resp.text):
                            app.logger.info("worker reported different in process prompts: worker="+resp.text+" orch="+str(workers[w]["load"])+". updated to worker reported load")
                            workers[w]["load"] = int(resp.text)
                    workers[w]["last_status_check"] = time.time()
                else:
                    app.logger.info("worker "+str(w)+" did not respond, removing")
                    del_workers.append(w)
            
            for d in del_workers:
                remove_worker(d)
        except Exception as ee:
            app.logger.info("worker monitor experienced and error ("+str(ee)+")")
        

def worker_ip(req):
    return request.environ.get('HTTP_X_REAL_IP', request.remote_addr)
    
def select_worker(sort_by):
    sort_workers = {}
    if sort_by == 'load':
        sort_workers = dict(sorted(workers.items(), key=lambda x: x[1]['load'], reverse=False))
    elif sort_by == 'score':
        sort_workers = dict(sorted(workers.items(), key=lambda x: x[1]['score'], reverse=True))
    elif sort_by == 'error_cnt':
        sort_workers = dict(sorted(workers.items(), key=lambda x: x[1]['error_cnt'], reverse=False))
    
    for w in sort_workers.keys():
        if sort_workers[w]['config']['maxsessions'] > sort_workers[w]['load']:
            id = sort_workers[w]['config']['id']
            workers[id]["load"] += 1
            return sort_workers[w]['config']
    
    app.logger.info("no worker selected, all workers busy")
    return None

def worker_done(id, iserror=False, resp_time=0):
    #update load to remove prompt
    workers[id]["load"] -= 1
    #record error
    if iserror:
        workers[id]["error_cnt"] += 1
    #track response time and score (based on 30 second response time)
    if resp_time > 0:
        if len(workers[id]["resp_time"]) == 10:
            workers[id]["resp_time"].pop(0)
        workers[id]["resp_time"].append(resp_time)
        if len(workers[id]["score"]) == 10:
            workers[id]["score"].pop(0)
        workers[id]["score"].append(30 / resp_time)

def remove_worker(id):
    global workers
    try:
        del workers[id]
        app.logger.info("worker "+id+" removed")
    except:
        app.logger.info("could not remove worker "+id+". worker id not found")

@app.route("/orchestratorconfig")
@manager
def orchestrator_config():
    return
    
@app.route("/workerstats", methods=['GET'])
@manager
def worker_stats():
    ws = copy.deepcopy(workers)
    for i in ws.keys():
        ws[i].pop("config",None)
    return jsonify(ws)
    
@app.route("/setworkerconfig/<id>")
@manager
def set_worker_config(id):
    token = request.values.get("token")
    sessions = request.values.get("sessions")
    gpu = request.values.get("gpu")
    mh = request.values.get("maxheight")
    mw = request.values.get("maxwidth")
    w_url = workers[id]["config"]["url"]
    
    if token != None:
        resp = sdr.post(w_url+"/accesstoken", data={'token':token})
        if resp.status_code != 200:
                return make_response(resp.content, resp.status_code)
    
    if sessions != None:
        resp = sdr.post(w_url+"/maxsessions", data={'sessions':sessions})
        if resp.status_code != 200:
            return make_response(resp.content, resp.status_code)
    
    if gpu != None:
        resp = sdr.post(w_url+"/gpu", data={'gpu':gpu})
        if resp.status_code != 200:
            return make_response(resp.content, resp.status_code)
    
    if mh != None or mw != None:
        h = 512
        w = 512
        if mh != None:
            h = mh
        if mw != None:
            w = mw
        resp = sdr.post(w_url+"/maximagesize", data={"maxheight":h,"maxwidth":w})
    
    resp = sdr.get(w_url+"/workerconfig")
    
    return make_response("worker config updated",200)

mw = threading.Timer(1, monitor_workers)
mw.daemon = True
def main(args):
    global config
    sdr.secret = args.secret
    sdr.verify_ssl = args.noselfsignedcert
    config["managesecret"] = args.managesecret
    
    #store worker monitor
    mw.start()
    
    app.logger.info("orchestrator config set, starting node")
    app.run(host=args.ipaddr, port=args.port)
     
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ipaddr", type=str, action="store", default="127.0.0.1")
    parser.add_argument("--port", type=str, action="store", default="5555")
    parser.add_argument("--secret", type=str, action="store", default="stablediffusion")
    parser.add_argument("--noselfsignedcert", type=bool, action="store", default=False)
    parser.add_argument("--managesecret", type=str, action="store", default="manage")
    args = parser.parse_args()
    
    main(args)