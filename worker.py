import sys, getopt, os, io, threading, hashlib, uuid, argparse, json, logging, traceback, time
from flask import Flask, request, send_file, make_response, jsonify
import sdhttp, sd
from PIL import Image
from PIL.PngImagePlugin import PngInfo

logging.basicConfig(level=logging.INFO)
outputs = {}
config={"url":"","id":uuid.uuid4(),"orchurl":"","orchcansetconfig":False,"maxsessions":1}

sdrequests = sdhttp.Sdrequests()
t = threading.BoundedSemaphore(1)


app = Flask(__name__)

@app.before_request 
def check_secret():
    if sdrequests.match_request_secret(request) == False:
        return make_response("secret does not match",500)
    
@app.route("/maxsessions", methods=['GET','POST'])
def max_sessions():
    if request.method == 'GET':
        return jsonify({"maxsessions":config["maxsessions"]}), 200
    if request.method == 'POST':
        sessions = request.values.get("sessions")
        if sessions.isnumeric():
            config["maxsessions"] = int(sessions)
            t = threading.BoundedSemaphore(int(sessions))
            return make_response("Ok", 200)
        else:
            return make_response("sessions not input as integer", 400)
    
@app.route("/accesstoken", methods=['GET','POST'])
def access_token():
    if request.method == 'GET':
        return jsonify({"accesstoken":sd.config["accesstoken"]}), 200
    if request.method == 'POST':
        token = request.values.get("token")
        if token[:2] == "hf" and len(token) == 37:
            sd.config["accesstoken"] = token
            load_model()
            return make_response("Ok", 200)
        else:
            return make_response("token not input correctly (start with 'hf' and be 37 characters", 400)

@app.route("/device", methods=['GET','POST'])
def gpu():
    if request.method == 'GET':
        return jsonify({"device":sd.settings["device"],"gpu":sd.settings["gpu"]}), 200
    if request.method == 'POST':
        device = request.values.get("device")
        gpu = request.values.get("gpu")
        if device == "cpu":
            sd.settings["device"] = "cpu"
        elif device == "cuda":
            if gpu.isnumeric():
                sd.settings["device"] = "cuda:"+str(gpu)
                sd.settings["gpu"] = int(gpu)
            else:
                sd.settings["device"] = "cuda:0"
                sd.settings["gpu"] = 0
        else:
            make_response("device not input correctly: device must be 'cpu' or 'cuda', gpu must be integer.")
        
        sd.pipe = sd.pipe.to(sd.settings["device"])
        return make_response("Ok", 200)

@app.route("/maximagesize", methods=['GET','POST'])
def max_image_size():
    if request.method == 'GET':
        return jsonify({"maxheight":sd.settings["maxheight"],"maxwidth":sd.settings["maxwidth"]})
    if request.method == 'POST':
        h = request.values.get("maxheight")
        w = request.values.get("maxwidth")
        
        if h.isnumeric() and w.isnumeric():
            sd.settings["maxheight"] = h
            sd.settings["maxwidth"] = w
            return make_response("max image size set", 200)
        else:
            return make_response("must input numeric height and width", 400)

@app.route("/workerstatus")
def send_status():
    return make_response("running",200)
    
@app.route("/workerconfig", methods=['GET'])
def send_worker_config():
    config = worker_config()
    return jsonify(config), 200

def worker_config():
    return {**config, **sd.settings}

@app.route("/txt2img", methods=['GET'])
def txt2img():
    prompt = request.values.get("prompt")
    if prompt == "":
        return make_response("prompt must be specified",400)
    
    app.logger.info("processing txt2img: " + prompt)
    
    guidance = float(request.values.get("guidance"))
    iterations = int(request.values.get("iterations"))
    height = int(request.values.get("height"))
    width = int(request.values.get("width"))
    batch_size = int(request.values.get("batch_size"))
    seed = request.values.get("seed").split(",")
    seed_step = int(request.values.get("seed_step"))
    prompt_id = request.headers.get("prompt_id")
    if prompt_id == "":
        prompt_id = str(uuid.uuid4())
    try:
        with t:
            #pipe returns [images] and if [nsfw_content_detected]
            images, nsfw, seeds, is_busy = sd.process_txt2img_prompt(prompt, guidance, iterations, height, width, batch_size, seed, seed_step)
    except ValueError as ve:
        return make_response("image processing busy, please re-submit", 503)
    
    if images != None:
        grid = image_grid(images,1,batch_size)
        grid_with_data = io.BytesIO()
        md = PngInfo()
        md.add_text("SD:prompt_id", prompt_id)
        md.add_text("SD:seeds", ",".join([str(s) for s in seeds]))
        grid.save(grid_with_data, pnginfo=md, format="png")
        grid_with_data.seek(0)
        return sdrequests.send_file_with_secret(send_file(grid_with_data, mimetype='image/png'))
    else:
        if is_busy == True:
            return make_response("image processing busy, please re-submit", 503)
        else:
            return make_response("image processing failed",500)
    
def image_grid(imgs, rows, cols):
    app.logger.info("making grid: imgs "+str(len(imgs))+" size "+str(cols))
    
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def register_with_orch():
    global config
    app.logger.info("registering with orchestrator: "+config["orchurl"])
    resp = sdrequests.post(config["orchurl"]+"/registerworker", json=worker_config())
    if resp.status_code == 200:
        app.logger.info("worker registered to orchestrator: "+config["orchurl"])
        return True
    elif resp.status_code == 400:
        app.logger.info("worker could not register, id already in use")
        return False
    else:
        app.logger.warning("worker could not register to orchestrator")
        return False
        
def monitor_worker_registered():
    global config
    while True:
        resp = sdrequests.get(config["orchurl"]+"/workerisregistered/"+config["id"])
        if resp.status_code == 400:
            app.logger.info("re-registering with orch")
            register_with_orch()
        elif resp.status_code == 404:
            app.logger.info("worker id already in use, enter a new worker id")
        time.sleep(30)

mw = threading.Timer(1, monitor_worker_registered)
mw.daemon = True
def main(args):
    global config
    sdrequests.secret = args.secret
    #set worker config
    config["url"] = "https://" + args.ipaddr + ":" + args.port
    if args.id != "":
        config["id"] = args.id
    if args.maxsessions > 1:
        t = threading.BoundedSemaphore(1)
    config["orchcansetconfig"] = args.orchcansetconfig
    #set stable diffusion config
    sd.settings["gpu"] = args.gpu
    sd.settings["lowermem"] = args.lowermem
    sd.settings["modelpath"] = args.modelpath
    sd.settings["maxheight"] = args.maxheight
    sd.settings["maxwidth"] = args.maxwidth
    sd.settings["accesstoken"] = args.accesstoken
    if args.device == "cuda":
        sd.settings["device"] = "cuda:"+str(args.gpu)
    
    app.logger.info("worker config set to:" + str(worker_config()))
    app.logger.info("stable diffusions settings are: " + str(sd.settings))
    #register worker with O if specified
    if args.orchurl != "":
        config["orchurl"] = args.orchurl
        if not "https://" in args.orchurl:
            config["orchurl"] = "https://"+config["orchurl"]
        if register_with_orch():
            #start worker registered monitor
            mw.start()
        else:
            return
    #load the model
    sd.load_model()
    
    if sd.pipe != None:
        app.logger.info("model loaded, starting web server")
        if args.ipaddr == "127.0.0.1":
            app.run(host="127.0.0.1", port=args.port, ssl_context="adhoc", threaded=True)
        else:
            app.run(host="0.0.0.0", port=args.port, ssl_context="adhoc", threaded=True)
    else:
        app.logger.info("model not loaded, exiting")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ipaddr", type=str, action="store", default="127.0.0.1")
    parser.add_argument("--port", type=str, action="store", default="5555")
    parser.add_argument("--secret", type=str, action="store", default="stablediffusion")
    parser.add_argument("--lowermem", type=bool, action="store", default=False)
    parser.add_argument("--maxheight", type=int, action="store", default=512)
    parser.add_argument("--maxwidth", type=int, action="store", default=512)
    parser.add_argument("--maxsessions", type=int, action="store", default=1)
    parser.add_argument("--accesstoken", type=str, action="store", default="")
    parser.add_argument("--modelpath", type=str, action="store", default="")
    parser.add_argument("--orchcansetconfig", type=bool, action="store", default=False)
    parser.add_argument("--id", type=str, action="store", default="")
    parser.add_argument("--orchurl", type=str, action="store", default="")
    parser.add_argument("--device", type=str, action="store", default=0, help="cpu or cuda")
    parser.add_argument("--gpu", type=int, action="store", default=0, help="select gpu to use with 'cuda' device, default is first gpu")
    args = parser.parse_args()
    
    main(args)
    
