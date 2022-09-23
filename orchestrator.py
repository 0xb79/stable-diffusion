from flask import Flask, request, send_file, Response, make_response, jsonify
from torch import autocast, cuda
from functools import wraps
import sys, os, io, logging, uuid, torch, requests, argparse, json, time, copy, threading
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
import sdhttp

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

config = {"sort_workers_by":"load"}
sdrequests = sdhttp.Sdrequests()
workers = {}


def credentials_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if "Credentials" in request.headers and sdrequests.match_request_secret(request):
            return f(*args, **kwargs)
        else:
            return make_response("secret does not match", 400)
    return wrap
    
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
    
    batch_size = request.values.get("batchsize")
    if batch_size != None:
        if batch_size.isnumeric() == False:
            return make_response("need to input numeric batchsize", 400)
    else:
        batch_size = 1
    
    seed = request.values.get("seed")
    if seed == None:
        seed = ''
    
    seed_step = request.values.get("seedstep")
    if seed_step != None:
        if seed_step.isnumeric() == False:
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
            resp = sdrequests.get(worker['url']+"/txt2img",headers={"prompt_id":prompt_id},params={"prompt":prompt,"batch_size":batch_size,"guidance":guidance,"iterations":iterations,"height":height,"width":width,"seed":seed, "seed_step":seed_step})
        
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
            return make_response("could not process prompt", 500)
    else:
        app.logger.info("no worker available")
        return make_response("no workers available",503)

@app.route("/workerstats", methods=['GET'])
@credentials_required
def worker_stats():
    ws = copy.deepcopy(workers)
    for i in ws.keys():
        ws[i].pop("config",None)
    return jsonify(ws)

@app.route("/registerworker", methods=['POST'])
@credentials_required
def register_worker():
    global workers
    w_config = request.get_json()
    w_ip = worker_ip(request)
    app.logger.info("worker registration received: "+str(w_config))
    if w_config["url"] == "":
        resp = sdrequests.make_response_with_secret(make_response('url must be set',404))
        return resp
    else:
        if w_config["id"] in workers.keys():
            if workers[w_config["id"]]["remote_addr"] != w_ip:
                app.logger.info("worker id ("+w_config["id"]+") already registered at different ip address")
                return sdrequests.make_response_with_secret(make_response('id already in use',400))
        
        workers[w_config["id"]] = {'config':w_config,'load':0,'score':[], 'resp_time':[], 'error_cnt':0, "remote_addr":w_ip, "last_checkin":0,"last_status_check":0}
        app.logger.info("worker registered  (id: "+w_config["id"]+")")
        resp = sdrequests.make_response_with_secret(make_response('worker registered',200))
        return resp

@app.route("/workerisregistered/<id>", methods=['GET'])
@credentials_required
def worker_is_registered(id):
    global workers
    w_ip = worker_ip(request)
    if id in workers.keys():
        if workers[id]["remote_addr"] == w_ip:
            workers[id]["last_checkin"] = time.time()
            return sdrequests.make_response_with_secret(make_response("",200))
        else:
            return sdrequests.make_response_with_secret(make_response("worker id already registered a differnet ip address",404))
    else:
        app.logger.info("worker "+id+" not registered, expecting registration request")
        return sdrequests.make_response_with_secret(make_response("",400))

def monitor_workers():
    global workers
    while True:
        time.sleep(10)
        del_workers=[]
        for w in workers.keys():
            resp = sdrequests.get(workers[w]["config"]["url"]+"/workerstatus")
            if resp.status_code == 200:
                if resp.text.isnumeric():
                    if workers[w]["load"] != int(resp.text):
                        app.logger.info("worker reported different in process prompts: worker="+resp.text+" orch="+workers[w]["load"]+". updated to worker reported load")
                        workers[w]["load"] = int(resp.text)
                workers[w]["last_status_check"] = time.time()
            else:
                app.logger.info("worker "+w+" did not respond, removing")
                del_workers.append(w)
        
        for d in del_workers:
            remove_worker(d)

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

@app.route("/setworkerconfig/<id>")
@credentials_required
def set_worker_config(id):
    token = request.values.get("token")
    sessions = request.values.get("sessions")
    gpu = request.values.get("gpu")
    mh = request.values.get("maxheight")
    mw = request.values.get("maxwidth")
    w_url = workers[id]["config"]["url"]
    
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
    
    resp = sdrequests.get(w_url+"/workerconfig")
    
    return make_response("worker config updated",200)

mw = threading.Timer(1, monitor_workers)
mw.daemon = True
def main(args):
    sdrequests.secret = args.secret
    sdrequests.verify_ssl = args.noselfsignedcert
    
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
    args = parser.parse_args()
    
    main(args)