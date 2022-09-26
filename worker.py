import sys, getopt, os, io, threading, hashlib, uuid, argparse, json, logging, traceback, time
from flask import Flask, request, send_file, make_response, jsonify
import sdhttp, sd
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from functools import wraps
from helpers import is_number

logging.basicConfig(level=logging.INFO)
outputs = {}
config={"url":"","id":uuid.uuid4(),"orchurl":"","orchcansetconfig":False,"maxsessions":1,"in_process":0,"txt2img":True,"img2img":True}

sdr = sdhttp.InternalRequests()
sdp = sd.StableDiffusionProcessor()
t = threading.BoundedSemaphore(1)


app = Flask(__name__)

def internal(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if "Credentials" in request.headers and sdr.match_request_secret(request):
            return f(*args, **kwargs)
        else:
            return make_response("secret does not match", 400)
    return wrap

@app.route("/maxsessions", methods=['GET','POST'])
@internal
def max_sessions():
    if request.method == 'GET':
        return jsonify({"maxsessions":config["maxsessions"]}), 200
    if request.method == 'POST':
        sessions = request.values.get("sessions")
        if is_number(sessions):
            config["maxsessions"] = int(sessions)
            t = threading.BoundedSemaphore(int(sessions))
            return sdp.make_response_with_secret("Ok", 200)
        else:
            return sdp.make_response_with_secret("sessions not input as integer", 400)
    
@app.route("/accesstoken", methods=['GET','POST'])
@internal
def access_token():
    if request.method == 'GET':
        return jsonify({"accesstoken":sd.config["accesstoken"]}), 200
    if request.method == 'POST':
        token = request.values.get("token")
        if token[:2] == "hf" and len(token) == 37:
            sd.config["accesstoken"] = token
            load_model()
            return sdp.make_response_with_secret("Ok", 200)
        else:
            return sdp.make_response_with_secret("token not input correctly (start with 'hf' and be 37 characters", 400)

@app.route("/device", methods=['GET','POST'])
@internal
def gpu():
    if request.method == 'GET':
        return jsonify({"device":sdp.settings["device"],"gpu":sdp.settings["gpu"]}), 200
    if request.method == 'POST':
        device = request.values.get("device")
        gpu = request.values.get("gpu")
        if device == "cpu":
            sdp.settings["device"] = "cpu"
        elif device == "cuda":
            if is_number(gpu):
                sdp.settings["device"] = "cuda:"+str(gpu)
                sdp.settings["gpu"] = int(gpu)
            else:
                sdp.settings["device"] = "cuda:0"
                sdp.settings["gpu"] = 0
        else:
            sdp.make_response_with_secret("device not input correctly: device must be 'cpu' or 'cuda', gpu must be integer.")
        
        try:
            sdp.load_model(config["txt2img"], config["img2img"])
            return sdp.make_response_with_secret("Ok", 200)
        except:
            return sdp.make_response_with_secret("could not change gpu selected", 500)

@app.route("/maximagesize", methods=['GET','POST'])
@internal
def max_image_size():
    if request.method == 'GET':
        return jsonify({"maxheight":sd.settings["maxheight"],"maxwidth":sd.settings["maxwidth"]})
    if request.method == 'POST':
        h = request.values.get("maxheight")
        w = request.values.get("maxwidth")
        
        if is_number(h) and is_number(w):
            sd.settings["maxheight"] = h
            sd.settings["maxwidth"] = w
            return sdp.make_response_with_secret("max image size set", 200)
        else:
            return sdp.make_response_with_secret("must input numeric height and width", 400)

@app.route("/workerstatus")
@internal
def send_status():
    return sdr.make_response_with_secret(str(config["in_process"]),200)
    
@app.route("/workerconfig", methods=['GET'])
@internal
def send_worker_config():
    config = worker_config()
    return jsonify(config), 200

def worker_config():
    return {**config, **sdp.settings}

@app.route("/txt2img", methods=['POST'])
@internal
def txt2img():
    global config
    prompt = request.values.get("prompt")
    if prompt == "":
        return sdp.make_response_with_secret("prompt must be specified",400)
    
    app.logger.info("processing txt2img: " + prompt)
    
    guidance = float(request.values.get("guidance"))
    iterations = int(request.values.get("iterations"))
    height = int(request.values.get("height"))
    width = int(request.values.get("width"))
    batch_size = int(request.values.get("batchsize"))
    seed = request.values.get("seed").split(",")
    seed_step = int(request.values.get("seedstep"))
    prompt_id = request.headers.get("prompt_id")
    resp = sdp.make_response_with_secret("",200)
    
    images = None
    seeds = ['']
    if prompt_id == "":
        prompt_id = str(uuid.uuid4())
    try:
        with t:
            #pipe returns [images] and if [nsfw_content_detected]
            config["in_process"] += 1
            images, seeds = sdp.process_txt2img_prompt(prompt, guidance, iterations, height, width, batch_size, seed, seed_step)
    except ValueError as ve:
        resp = sdp.make_response_w_secret("image processing busy, please resubmit", 503)
    except sd.ModelLoadingError as me:
        resp = sdp.make_response_with_secret("model loading, please resubmit", 503)
    except sd.ProcessingError as pe:
        resp = sdp.make_response_with_secret("image processing had error", 500)
    finally:
        config["in_process"] -= 1
        if resp.status_code != 200:
            return resp
    
    if images != None:
        grid = image_grid(images, 1, batch_size)
        grid_with_data = io.BytesIO()
        md = PngInfo()
        md.add_text("SD:prompt", prompt)
        md.add_text("SD:prompt_id", prompt_id)
        md.add_text("SD:seeds", ",".join([str(s) for s in seeds]))
        grid.save(grid_with_data, pnginfo=md, format="png")
        grid_with_data.seek(0)
        return sdr.send_file_with_secret(grid_with_data, addl_headers={"prompt_id":prompt_id})
    else:
        return sdp.make_response_with_secret("image processing failed",500)
        
@app.route("/img2img", methods=['POST'])
@internal
def img2img():
    prompt = request.values.get("prompt")
    if prompt == "":
        return sdp.make_response_with_secret("prompt must be specified",400)
    
    init_img = process_init_img(io.BytesIO(request.files.get("init_img")))
    if img == None:
        return sdp.make_response_with_secret("no init_img provided", 400)
        
    guidance = float(request.values.get("guidance"))
    strength = float(request.values.get("strength"))
    iterations = int(request.values.get("iterations"))
    batch_size = int(request.values.get("batchsize"))
    seed = request.values.get("seed")
    
    try:
        with t:
            config["in_process"] += 1
            images = sdp.process_img2img_prompt(prompt, init_img, guidance, strength, iterations, batch_size, seed)
    except ValueError as ve:
        resp = sdp.make_response_with_secret("image processing busy, please resubmit", 503)
    except sd.ModelLoadingError as me:
        resp = sdp.make_response_with_secret("model loading, please resubmit", 503)
    except sd.ProcessingError as pe:
        resp = sdp.make_response_with_secret("image processing had error", 500)
    finally:
        config["in_process"] -= 1
        if resp.status_code != 200:
            return resp
    
    if images != None:
        grid = image_grid(images, 1, batch_size)
        grid_with_data = io.BytesIO()
        md = PngInfo()
        md.add_text("SD:prompt", prompt)
        md.add_text("SD:prompt_id", prompt_id)
        md.add_text("SD:seed", seed)
        grid.save(grid_with_data, pnginfo=md, format="png")
        grid_with_data.seek(0)
        return sdr.send_file_with_secret(grid_with_data, addl_headers={"prompt_id":prompt_id})
    else:
        return sdp.make_response_with_secret("image processing failed", 500)

def image_grid(imgs, rows, cols):
    app.logger.info("making grid: imgs "+str(len(imgs))+" size "+str(cols))
    
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid
    
def process_init_img(init_img):
    init_img = Image.open(init_img).convert("RGB")
    return init_img.thumbnail((sdp.settings["maxwidth"], sdp.settings["maxheight"]))
    
def register_with_orch():
    global config
    app.logger.info("registering with orchestrator: "+config["orchurl"])
    resp = sdr.post(config["orchurl"]+"/registerworker", json=worker_config())
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
        resp = sdr.get(config["orchurl"]+"/workerisregistered/"+config["id"])
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
    sdr.secret = args.secret
    #set worker config
    config["url"] = "https://" + args.ipaddr + ":" + args.port
    if args.id != "":
        config["id"] = args.id
    if args.maxsessions > 1:
        t = threading.BoundedSemaphore(args.maxsessions)
    config["orchcansetconfig"] = args.orchcansetconfig
    config["txt2img"] = args.txt2img
    config["img2img"] = args.img2img
    #set stable diffusion config
    sdp.settings["gpu"] = args.gpu
    sdp.settings["lowermem"] = args.lowermem
    sdp.settings["slicemem"] = args.slicemem
    sdp.settings["modelpath"] = args.modelpath
    sdp.settings["maxheight"] = args.maxheight
    sdp.settings["maxwidth"] = args.maxwidth
    sdp.settings["accesstoken"] = args.accesstoken
    if args.device == "cuda":
        sdp.settings["device"] = "cuda:"+str(args.gpu)
    
    app.logger.info("worker config set to:" + str(config))
    app.logger.info("stable diffusions settings are: " + str(sdp.settings))
    
    #load the model
    app.logger.info("loading model to "+sdp.settings["device"])
    sdp.load_model(args.txt2img, args.img2img)
    t2i_model_loaded, i2i_model_loaded = sdp.models_are_loaded()
    if args.txt2img:
        if t2i_model_loaded:
            app.logger.info("txt2img model loaded")
        else:
            app.logger.info("txt2img model not loaded, exiting")
            return
        
    if args.img2img:
        if i2i_model_loaded:
            app.logger.info("img2img model loaded")
        else:
            app.logger.info("img2img model not loaded, exiting")
            return
    
    #register worker with O if specified
    if args.orchurl != "":
        config["orchurl"] = args.orchurl
        if not "https://" in args.orchurl:
            config["orchurl"] = "https://"+config["orchurl"]
        if register_with_orch():
            #start worker registered monitor
            mw.start()
        else:
            app.logger.info("could not register with orchestrator, exiting")
            return
        
    app.logger.info("model loaded, starting web server")
    if args.ipaddr == "127.0.0.1":
        app.run(host="127.0.0.1", port=args.port, ssl_context="adhoc", threaded=True)
    else:
        app.run(host="0.0.0.0", port=args.port, ssl_context="adhoc", threaded=True)



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ipaddr", type=str, action="store", default="127.0.0.1")
    parser.add_argument("--port", type=str, action="store", default="5555")
    parser.add_argument("--secret", type=str, action="store", default="stablediffusion")
    parser.add_argument("--slicemem", action="store_true")
    parser.add_argument("--lowermem", action="store_true")
    parser.add_argument("--txt2img", action="store_true")
    parser.add_argument("--img2img", action="store_true")
    parser.add_argument("--maxheight", type=int, action="store", default=512)
    parser.add_argument("--maxwidth", type=int, action="store", default=512)
    parser.add_argument("--maxsessions", type=int, action="store", default=1)
    parser.add_argument("--accesstoken", type=str, action="store", default="")
    parser.add_argument("--modelpath", type=str, action="store", default="")
    parser.add_argument("--orchcansetconfig", action="store_true")
    parser.add_argument("--id", type=str, action="store", default="")
    parser.add_argument("--orchurl", type=str, action="store", default="")
    parser.add_argument("--device", type=str, action="store", default=0, help="cpu or cuda")
    parser.add_argument("--gpu", type=int, action="store", default=0, help="select gpu to use with 'cuda' device, default is first gpu")
    args = parser.parse_args()
    
    main(args)
    
