import sys, getopt, os, threading, hashlib, torch, uuid, argparse
from flask import Flask, request, send_file, make_response, jsonify
from PIL.PngImagePlugin import PngInfo
from PIL import Image
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
import sdhttp, logger

app = Flask(__name__)

url = ""
id = uuid.uuid4()
access_token = "enter access token"
model_path = ""
lower_mem = False
max_height = 512
max_width = 512
max_sessions = 1
gpu = 0
orch_url = ""
orch_can_set_config = False

max_batch_size = 32
outputs = {}
pipe = None
lms = None

sdrequests = sdhttp.Sdrequests()
t = threading.BoundedSemaphore(max_sessions)

@app.before_request 
def check_secret():
    if sdrequests.match_request_secret(request) == False:
        return make_response("secret does not match",500)
        
@app.route("/accesstoken", methods=['GET','POST'])
def access_token():
    if request.method == 'GET':
        return jsonify({"access_token":access_token}), 200
    if request.method == 'POST':
        token = request.values.get("token")
        if token[:2] == "hf" and len(token) == 37:
            access_token = token
            load_model()
            return make_response("Ok", 200)
        else:
            return make_response("token not input correctly (start with 'hf' and be 37 characters", 400)

@app.route("/maxsessions", methods=['GET','POST'])
def max_sessions():
    if request.method == 'GET':
        return jsonify({"max_sessions":max_sessions}), 200
    if request.method == 'POST':
        sessions = request.values.get("sessions")
        if sessions.isnumeric():
            max_sessions = int(sessions)
            t = threading.BoundedSemaphore(int(sessions))
            return make_response("Ok", 200)
        else:
            return make_response("sessions not input as integer", 400)

@app.route("/gpu", methods=['GET','POST'])
def gpu():
    if request.method == 'GET':
        return jsonify({"gpu":gpu}), 200
    if request.method == 'POST':
        gpu = request.values.get("gpu")
        if gpu.isnumeric():
            gpu = int(gpu)
            pipe = pipe.to("cuda:"+gpu)
            return make_response("Ok", 200)
        else:
            return make_response("gpu not input as integer", 400)

@app.route("/maximagesize", methods=['GET','POST'])
def max_image_size():
    if request.method == 'GET':
        return jsonify({"max_height":max_height,"max_width":max_width})
    if request.meoth == 'POST':
        h = request.values.get("maxheight")
        w = request.values.get("maxwidth")
        
        if h.isnumeric() and w.isnumeric():
            max_height = h
            max_width = w
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
    config = {"maxheight": max_height, "maxwidth": max_width, "maxsessions": max_sessions, "gpu": gpu, "lowermem": lower_mem, "url":url, "id":id}
    return config

@app.route("/txt2img", methods=['GET'])
def txt2img():
    prompt = request.values.get("prompt")
    prompt = prompt.replace(" ","-")
    prompt = prompt.replace(".", "")
    prompt = prompt.replace("/","")
    
    app.logger.info("processing txt2img: " + prompt)
    
    guidance = prompt.values.get("guidance")
    iterations = prompt.values.get("iterations")
    height = prompt.values.get("height")
    width = prompt.values.get("width")
    batch_size = prompt.values.get("batch_size")
    seed = prompt.values.get("seed").split(",")
    seed_step = prompt.values.get("seed_step")
    prompt_id = request.headers.get("prompt_id")
    
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
        
        model = "CompVis/stable-diffusion-v1-4" if model_path == "" else model_path
        if lower_mem == False:
            pipe = StableDiffusionPipeline.from_pretrained(
                model, 
                scheduler=lms,
                use_auth_token=access_token
            )
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                model, 
                revision="fp16", 
                torch_dtype=torch.float16,
                scheduler=lms,
                use_auth_token=access_token
            )
        
        if torch.cuda.is_available():
            app.logger.info("using cuda")
            pipe = pipe.to("cuda:1")
            if lower_mem:
                app.logger.info("pipe set to use less gpu memory")
                pipe.enable_attention_slicing()
        
    except:
        app.logger.warning("could not load model, make sure access token or model_path is set")
        pipe = None
        lms = None
        
def process_txt2img_prompt(prompt='', guidance=7.5, iterations=50, height=512, width=512, batch_size=1, seed='', seed_step=0):
    height = min(max_height, int(height))
    width = min(max_width, int(width))
    batch_size = min(max_batch_sie, int(batch_size))
    
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
    
    
    url = "https://" + args.ipaddr + ":" + args.port
    sdrequests.secret = args.secret
    gpu = args.gpu
    lower_mem = args.lowermem
    model_path = args.modelpath
    max_height = args.maxheight
    max_width = args.maxwidth
    orch_can_set_config = args.orchcansetconfig
    if args.id != "":
        id = args.id
    
    #register worker with O if specified
    if args.orchurl != "":
        orch_url = args.orchurl
        if not "https://" in args.orch_url:
            orch_url = "https://"+orch_url
        
        resp = sdrequests.post(orch_url+"/registerworker", json=worker_config())
        if resp.status_code == 200:
            app.logger.info("worker registered to orchestrator: "+orch_url)
        else:
            app.logger.warning("worker could not register to orchestrator")
    
    app.run(host=ip, port=p, ssl_context="adhoc", threaded=True)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ipaddr", type=str, action="store", default="127.0.0.1")
    parser.add_argument("--port", type=str, action="store", default="5555")
    parser.add_argument("--secret", type=str, action="store", default="stablediffusion")
    parser.add_argument("--lowermem", type=bool, action="store", default=False)
    parser.add_argument("--maxheight", type=int, action="store", default=512)
    parser.add_argument("--maxwidth", type=int, action="store", default=512)
    parser.add_argument("--modelpath", type=str, action="store", default="")
    parser.add_argument("--orchcansetconfig", type=bool, action="store", default=False)
    parser.add_argument("--id", type=str, action="store", default="")
    parser.add_argument("--orchurl", type=str, action="store", default="")
    parser.add_argument("--gpu", type=int, action="store", default=0)
    args = parser.parse_args()
    
    main(args)
    
