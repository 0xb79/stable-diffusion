from flask import Flask, request, send_file, Response, make_response, jsonify
from flask_executor import Executor
from torch import autocast, cuda
from functools import wraps
from PIL import Image
import sys, os, io, logging, uuid, torch, requests, argparse, json, time, copy, threading, asyncio, datetime, pathlib
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
import sdhttp
from helpers import is_number

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s',level=logging.INFO)
logger = logging.getLogger(__name__)

config = {"sort_workers_by":"load","managesecret":"","maxsessions":10,"datadir":""}
sdr = sdhttp.InternalRequests()
workers = {}

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000 #limit to 16 megabytes

root_dir = os.path.dirname(os.path.abspath(__file__))

def internal(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if "Credentials" in request.headers and sdr.match_request_secret(request):
            return f(*args, **kwargs)
        else:
            return make_response("secret does not match", 401)
    return wrap
    
def manager(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if "Credentials" in request.headers and request.headers["Credentials"] == config["managesecret"]:
            return f(*args, **kwargs)
        else:
            return make_response("secret does not match", 401)
    return wrap
    
def timed(f):
    @wraps(f)
    def wrap(*a, **k):
        then = time.time()
        res = f(*a, **k)
        elapsed = time.time() - then
        return elapsed, res
    return wrap

@app.route("/txt2img", methods=['GET','POST'])
def start_process_txt2img():
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
    
    batchsize = request.values.get("batchsize")
    if batchsize != None:
        if is_number(batchsize) == False:
            return make_response("need to input numeric batchsize", 400)
    else:
        batchsize = 1
    
    seed = request.values.get("seed")
    if seed == None:
        seed = ''
    
    seedstep = request.values.get("seedstep")
    if seedstep != None:
        if is_number(seedstep) == False:
            return make_response("need to input numeric seedstep", 400)
        else:
            seedstep = 1
    else:
        seedstep = 1
    
    
    #select worker to send to
    if len(workers) == 0:
        return make_response("no workers registered", 500)
        
    req = request.full_path
    prompt_id = str(uuid.uuid4())
    worker = select_worker(config["sort_workers_by"], prompt_id)
    if worker != None:
        app.logger.info("worker selected: "+worker["id"])
        resp = sdr.post(url=worker['url']+"/txt2img", headers={"prompt_id":prompt_id}, data={"prompt":prompt,"batchsize":batchsize,"guidance":guidance,"iterations":iterations,"height":height,"width":width,"seed":seed, "seedstep":seedstep})
        if resp.status_code == 200:
            app.logger.info(prompt_id+": txt2img: worker "+worker["id"]+" accepted prompt")
            return make_response(prompt_id, 200)
        elif resp.status_code == 503:
            app.logger.info(prompt_id+": txt2img: worker "+worker["id"]+" capped")
            return make_response("no workers available", 503)
        else:
            app.logger.info(prompt_id+": txt2img: worker "+worker["id"]+" error")
            return make_response("error from worker", 500)
    else:
        app.logger.info("no worker available")
        return make_response("no workers available", 503)
    
@app.route("/img2img", methods=['GET', 'POST'])
def start_process_img2img():
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
    
    batchsize = request.values.get("batchsize")
    if batchsize != None:
        if is_number(batchsize) == False:
            return make_response("need to input numeric batchsize", 400)
    else:
        batchsize = 1
    
    #select worker to send to
    if len(workers) == 0:
        return make_response("no workers registered", 500)
        
    req = request.full_path
    prompt_id = str(uuid.uuid4())
    worker = select_worker(config["sort_workers_by"], prompt_id)
    if worker != None:
        app.logger.info("worker selected: "+worker["id"])
        resp = sdr.post(url=worker['url']+"/img2img",headers={"prompt_id":prompt_id},params={"prompt":prompt,"batchsize":batchsize,"guidance":guidance,"strength":strength,"iterations":iterations, "seed":seed})
        if resp.status_code == 200:
            app.logger.info(prompt_id+": img2img: worker "+worker["config"]["id"]+" accepted prompt")
            return make_response(prompt_id, 200)
        elif resp.status_code == 503:
            app.logger.info(prompt_id+": img2img: worker "+worker["config"]["id"]+" capped")
            return make_response("no workers available", 503)
        else:
            app.logger.info(prompt_id+": img2img: worker "+worker["config"]["id"]+" error")
            return make_response("error from worker", 500)
    else:
        app.logger.info("no worker available")
        return make_response("no workers available", 503)
        

@app.route("/results/<prompt_id>", methods=['GET'])
def send_results(prompt_id):
    pf = prompt_results_file(prompt_id)
    if os.path.isfile(pf):
        return send_file(pf, mimetype='image/png')
    else:
        #return 
        for w in workers:
            for p in workers[w][load]:
                if p[0] == prompt_id:
                    return make_response("prompt results not available, still in process", 204)
        #prompt not available and not in process
        return make_response("prompt results not available", 404)

@internal
@app.route("/resultsfromworker/<prompt_id>", methods=['POST'])
def results_from_worker(prompt_id):
    if len(request.files) > 0:
        try:
            worker_done(prompt_id, time.time(), False)
            img = Image.open(request.files['img'].stream)
            #img.verify()
            save_to = prompt_results_file(prompt_id)
            app.logger.info("saving image to: "+save_to)
            img.save(prompt_results_file(prompt_id), format="png")
            app.logger.info(prompt_id+": results received and saved")
            return sdr.send_response_with_secret("results received", 200)
        except Exception as ee:
            app.logger.error("error", exc_info=True)
            return sdr.send_response_with_secret("image is not valid", 400)
    else:
        app.logger.info(prompt_id+": no files received")
        worker_done(prompt_id, time.time(), True)  #no file received, worker had error
        return sdr.send_response_with_secret("no files attached", 400)

def prompt_results_file(prompt_id):
    return os.path.join(config["datadir"],prompt_id+".png")
    
@internal
@app.route("/registerworker", methods=['POST'])
def register_worker():
    global workers
    w_config = request.get_json()
    w_ip = worker_ip(request)
    app.logger.info("worker registration received: "+str(w_config))
    if w_config["url"] != "":
        if w_config["id"] in workers.keys():
            if workers[w_config["id"]]["remote_addr"] != w_ip:
                app.logger.info("worker id ("+w_config["id"]+") already registered at different ip address")
                return sdr.send_response_with_secret('id already in use', 404)
        
        workers[w_config["id"]] = {'config':w_config,'load':[],'score':[], 'resp_time':[], 'error_cnt':0, "remote_addr":w_ip, "last_checkin":0,"last_status_check":0}
        app.logger.info("worker registered  (id: "+w_config["id"]+")")
        resp = sdr.send_response_with_secret('worker registered', 200)
        return resp
    else:
        resp = sdr.send_response_with_secret('url must be set', 400)
        return resp

@internal
@app.route("/workerisregistered/<id>", methods=['GET'])
def worker_is_registered(id):
    global workers
    w_ip = worker_ip(request)
    if id in workers.keys():
        if workers[id]["remote_addr"] == w_ip:
            workers[id]["last_checkin"] = time.time()
            return sdr.send_response_with_secret("",200)
        else:
            return sdr.send_response_with_secret("worker id already registered with a differnet ip address",404)
    else:
        app.logger.info("worker "+str(id)+" not registered, expecting registration request")
        return sdr.send_response_with_secret("",400)

def monitor_workers():
    global workers
    while True:
        time.sleep(10)
        try:
            del_workers=[]
            for w in workers.keys():
                resp = sdr.get(workers[w]["config"]["url"]+"/workerstatus")
                if resp.status_code == 200:
                    workers[w]["last_status_check"] = time.time()
                    for p in workers[w]["load"]:
                        start = p[1]
                        #remove prompt process if not responsed in 20 minutes
                        if time.time() > (start + 1200):
                            prompt_id = p[0]
                            app.logger.info(prompt_id+": did not return results in 20 minutes, removing")
                            workers[w]["load"].pop(p)
                else:
                    app.logger.info("worker "+str(w)+" did not respond, removing")
                    del_workers.append(w)
                    #TODO: add job resubmit logic if a worker went offline with jobs in process
                    
            for d in del_workers:
                remove_worker(d)
        except Exception as ee:
            app.logger.info("worker monitor experienced and error ("+str(ee)+")")
        

def worker_ip(req):
    return request.environ.get('HTTP_X_REAL_IP', request.remote_addr)
    
def select_worker(sort_by, prompt_id):
    global workers
    sort_workers = {}
    if sort_by == 'load':
        sort_workers = dict(sorted(workers.items(), key=lambda x: len(x[1]['load']), reverse=False))
    elif sort_by == 'score':
        sort_workers = dict(sorted(workers.items(), key=lambda x: x[1]['score'], reverse=True))
    elif sort_by == 'error_cnt':
        sort_workers = dict(sorted(workers.items(), key=lambda x: x[1]['error_cnt'], reverse=False))
    
    for w in sort_workers.keys():
        if sort_workers[w]['config']['maxsessions'] > len(sort_workers[w]['load']):
            id = sort_workers[w]['config']['id']
            workers[id]["load"].append((prompt_id,time.time()))
            return sort_workers[w]['config']
    
    app.logger.info("no worker selected, all workers busy")
    return None

def worker_done(prompt_id, rec_time=0, iserror=False):
    #update load to remove prompt
    global workers
    for w in workers.keys():
        for idx, job in enumerate(workers[w]["load"]):
            if job[0] == prompt_id:
                job = workers[w]["load"].pop(idx)
                p_start = job[1]
                #record error
                if iserror:
                    workers[w]["error_cnt"] += 1
                #track response time and score (based on 30 second response time)
                #TODO: calculate score based on image size * batchsize requested
                if rec_time > 0:
                    if len(workers[w]["resp_time"]) == 10:
                        workers[w]["resp_time"].pop(0)
                    resp_time = rec_time - p_start
                    workers[w]["resp_time"].append(resp_time)
                    if len(workers[w]["score"]) == 10:
                        workers[w]["score"].pop(0)
                        workers[w]["score"].append(30 / resp_time)
                return
        

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
    #check that data dir exists and create if not
    config["datadir"] = str(args.datadir)
    try:
        args.datadir.mkdir(parents=True, exist_ok=True)
    except:
        app.logger.warning("datadir does not exist and could not create. check permissions are correct. exiting.")
        return
    #set configs
    sdr.secret = args.secret
    sdr.verify_ssl = args.noselfsignedcert
    sdr.setup_workers(args.maxsessions)
    config["managesecret"] = args.managesecret
    config["maxsessions"] = args.maxsessions
    
    #store worker monitor
    mw.start()
    
    app.logger.info("orchestrator config set, starting node")
    app.run(host=args.ipaddr, port=args.port, threaded=True)
     
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ipaddr", type=str, action="store", default="127.0.0.1")
    parser.add_argument("--port", type=str, action="store", default="5555")
    parser.add_argument("--secret", type=str, action="store", default="stablediffusion")
    parser.add_argument("--datadir", type=lambda p: pathlib.Path(p).resolve(), default=pathlib.Path(__file__).resolve().parent / "odata")
    parser.add_argument("--noselfsignedcert", type=bool, action="store", default=False)
    parser.add_argument("--managesecret", type=str, action="store", default="manage")
    parser.add_argument("--maxsessions", type=int, action="store", default=10)
    args = parser.parse_args()
    
    main(args)