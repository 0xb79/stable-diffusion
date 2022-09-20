import sys, getopt, os, threading, hashlib, torch, uuid, argparse, json, logging
from flask import Flask, request, send_file, make_response, jsonify
from PIL.PngImagePlugin import PngInfo
from PIL import Image
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
import sdhttp

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

config = {"url":"","id":uuid.uuid4(),"accesstoken":"","modelpath":"","lowermem":False,"maxheight":512,"maxwidth":512,"maxsessions":1,"gpu":0,"orchurl":"","orchcansetconfig":False, "maxbatchsize":32}

outputs = {}
pipe = None
lms = None

sdrequests = sdhttp.Sdrequests()
t = threading.BoundedSemaphore(1)

@app.before_request 
def check_secret():
    if sdrequests.match_request_secret(request) == False:
        return make_response("secret does not match",500)
        
@app.route("/accesstoken", methods=['GET','POST'])
def access_token():
    if request.method == 'GET':
        return jsonify({"accesstoken":config["accesstoken"]}), 200
    if request.method == 'POST':
        token = request.values.get("token")
        if token[:2] == "hf" and len(token) == 37:
            config["accesstoken"] = token
            load_model()
            return make_response("Ok", 200)
        else:
            return make_response("token not input correctly (start with 'hf' and be 37 characters", 400)

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

@app.route("/gpu", methods=['GET','POST'])
def gpu():
    if request.method == 'GET':
        return jsonify({"gpu":config["gpu"]}), 200
    if request.method == 'POST':
        gpu = request.values.get("gpu")
        if gpu.isnumeric():
            config["gpu"] = int(gpu)
            pipe = pipe.to("cuda:"+gpu)
            return make_response("Ok", 200)
        else:
            return make_response("gpu not input as integer", 400)

@app.route("/maximagesize", methods=['GET','POST'])
def max_image_size():
    if request.method == 'GET':
        return jsonify({"maxheight":config["maxheight"],"maxwidth":config["maxwidth"]})
    if request.method == 'POST':
        h = request.values.get("maxheight")
        w = request.values.get("maxwidth")
        
        if h.isnumeric() and w.isnumeric():
            config["maxheight"] = h
            config["maxwidth"] = w
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
    return config

@app.route("/txt2img", methods=['GET'])
def txt2img():
    prompt = request.values.get("prompt")
    if prompt == "":
        return make_response("prompt must be specified",400)
    
    app.logger.info("processing txt2img: " + prompt)
    
    guidance = request.values.get("guidance")
    iterations = request.values.get("iterations")
    height = request.values.get("height")
    width = request.values.get("width")
    batch_size = request.values.get("batch_size")
    seed = request.values.get("seed").split(",")
    seed_step = request.values.get("seed_step")
    prompt_id = request.headers.get("prompt_id")
    if prompt_id == "":
        prompt_id = str(uuid.uuid4())
    
    #pipe returns [images] and if [nsfw_content_detected] 
    images, nsfw, seeds = process_txt2img_prompt(prompt, guidance, iterations, height, width, batch_size, seed, seed_step)
    
    grid = image_grid(images,1,batch_size)
    with io.BytesIO() as grid_with_data:
        md = PngInfo()
        md.add_text("SD:prompt_id",prompt_id)
        md.add_text("SD:seeds",seeds)
        grid.save(grid_with_data, pnginfo=md)
        return send_file(grid_with_data, mimetype='image/png',download_name=prompt_id+".png")
    
def load_model():
    try:
        pipe = None
        lms = None
        # this will substitute the default PNDM scheduler for K-LMS  
        lms = LMSDiscreteScheduler(
            beta_start=0.00085, 
            beta_end=0.012, 
            beta_schedule="scaled_linear"
        )
        
        model = "CompVis/stable-diffusion-v1-4" if config["modelpath"] == "" else config["modelpath"]
        if config["lowermem"] == False:
            pipe = StableDiffusionPipeline.from_pretrained(
                model, 
                scheduler=lms,
                use_auth_token=config["accesstoken"]
            )
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                model, 
                revision="fp16", 
                torch_dtype=torch.float16,
                scheduler=lms,
                use_auth_token=config["accesstoken"]
            )
        
        if torch.cuda.is_available():
            app.logger.info("loading model to cuda gpu "+str(config["gpu"]))
            pipe = pipe.to("cuda:"+str(config["gpu"]))
            if config["lowermem"]:
                app.logger.info("pipe set to use less gpu memory")
                pipe.enable_attention_slicing()
        
    except Exception as e: 
        print(e)
        app.logger.warning("could not load model, make sure accesstoken or modelpath is set")
        pipe = None
        lms = None
        
def process_txt2img_prompt(prompt='', guidance=7.5, iterations=50, height=512, width=512, batch_size=1, seed='', seed_step=0):
    height = min(config["maxheight"], int(height))
    width = min(config["maxwidth"], int(width))
    batch_size = min(config["maxbatchsize"], int(batch_size))
    
    if prompt == ['']:
        return make_response("must input prompt",400)
    try:
        with t:
            prompt = [prompt] * batch_size
            if torch.cuda.is_available():
                with torch.autocast("cuda"):
                    latents, seeds = create_latents(seed, batch_size, seed_step)
                    output = pipe(prompt, guidance=guidance, iterations=iterations, height=height, width=width, latents=latents)
                    return output.images, output.nsfw_content_detected, seeds
            else:
                latents, seeds = create_latents(seed, batch_size, seed_step)
                output = pipe(prompt, guidance=guidance, iterations=iterations, height=height, width=width)
                return output.images, output.nsfw_content_detected, seeds
            
        torch_gc()
    except ValueError:
        return make_response("image processing busy, please re-submit", 503)

def create_latents(seed='', batch_size=1, seed_step=0):
    generator = torch.Generator(device=device)
    latents = None
    seeds = []
    
    if seed != '':
        for s in seed:
            generator.manual_seed(seed)
            seeds.append(seed)
        else:
            seed = generator.seed()
            seeds.append(seed)
            image_latents = torch.randn(
                (1, pipe.unet.in_channels, height // 8, width // 8),
                generator = generator,
                device = device
            )
            latents = image_latents if latents is None else torch.cat((latents, image_latents))
    if batch_size > len(seed):
        addl = batch_size - len(seed)
        for _ in range(addl):
            if batch_step == 0:
                # Get a new random seed, store it and use it as the generator state
                seed = generator.seed()
                seeds.append(seed)
            else:
                #update the seed by the step
                seed = generator.manual_seed(seed+seed_step)
                seeds.append(seed)
            
            image_latents = torch.randn(
                (1, pipe.unet.in_channels, height // 8, width // 8),
                generator = generator,
                device = device
            )
            latents = image_latents if latents is None else torch.cat((latents, image_latents))
    
    return latents, seeds

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols
    
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid
    
def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    
#def root_dir():
#    return os.path.abspath(os.path.dirname(__file__))
    
def main(args):
    
    sdrequests.secret = args.secret
    config["url"] = "https://" + args.ipaddr + ":" + args.port
    config["gpu"] = args.gpu
    config["lowermem"] = args.lowermem
    config["modelpath"] = args.modelpath
    config["maxheight"] = args.maxheight
    config["maxwidth"] = args.maxwidth
    config["orchcansetconfig"] = args.orchcansetconfig
    config["accesstoken"] = args.accesstoken
    if args.id != "":
        config["id"] = args.id
    if args.maxsessions > 1:
        t = threading.BoundedSemaphore(1)
    
    print(worker_config())
    app.logger.info("worker config set to:" + str(worker_config()))
    #register worker with O if specified
    if args.orchurl != "":
        config["orchurl"] = args.orchurl
        if not "https://" in args.orchurl:
            config["orchurl"] = "https://"+config["orchurl"]
        
        resp = sdrequests.post(config["orchurl"]+"/registerworker", json=worker_config())
        if resp.status_code == 200:
            app.logger.info("worker registered to orchestrator: "+config["orchurl"])
        else:
            app.logger.warning("worker could not register to orchestrator")
    
    load_model()
    
    if pipe != None:
        app.logger.info("model loaded, starting web server")
        app.run(host=args.ipaddr, port=args.port, ssl_context="adhoc", threaded=True)
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
    parser.add_argument("--gpu", type=int, action="store", default=0)
    args = parser.parse_args()
    
    main(args)
    
