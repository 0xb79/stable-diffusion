import sys, getopt, os, threading, hashlib, torch
from flask import Flask, request, send_file, make_response, jsonify
from PIL.PngImagePlugin import PngInfo
from PIL import Image
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
import sdhttp

app = Flask(__name__)

access_token = "enter access token"
model_path = ""
lower_mem = False
max_height = 512
max_width = 512
max_sessions = 2
gpu = 0
outputs = {}
pipe = None
lms = None
t_id = uuid.uuid4()
orch_uri = ""

sdrequests = sdhttp.Sdrequests()
t = threading.BoundedSemaphore(max_sessions)

def load_model():
    try:
        # this will substitute the default PNDM scheduler for K-LMS  
        lms = LMSDiscreteScheduler(
            beta_start=0.00085, 
            beta_end=0.012, 
            beta_schedule="scaled_linear"
        )

        pipe = None
        model = "CompVis/stable-diffusion-v1-4" if model_path == "" else model_path
        if low_mem == False:
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
            print("using cuda")
            pipe = pipe.to("cuda:1")
            if low_mem:
                pipe.enable_attention_slicing()
        
    except:
        print("could not load model, make sure access token or model_path is set")
        pipe = None
        lms = None

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

@app.route("/transcoderconfig", methods=['GET'])
def transcoder_config():
    config = {"maxheight": max_height, "maxwidth": max_width, "max_sessions": max_sessions, "gpu": gpu, "max_sessions": max_sessions, "lowermem": lower_mem}
    return jsonify(config), 200

@app.route("/txt2img", methods=['GET'])
def get_result():
    if sqrequests.match_request_secret(request) == False:
        return make_response("secret does not match",500)
        
    prompt = request.values.get("prompt")
    prompt = prompt.replace(" ","-")
    prompt = prompt.replace(".", "")
    prompt = prompt.replace("/","")
    print(prompt)
    guidance = prompt.values.get("guidance")
    iterations = prompt.values.get("iterations")
    height = prompt.values.get("height")
    width = prompt.values.get("width")
    batch_size = prompt.values.get("batch_size")
    seed = prompt.values.get("seed").split(",")
    seed_step = prompt.values.get("seed_step")
    prompt_id = request.headers.get("prompt_id")
    
    #pipe returns [images] and if [nsfw_content_detected] 
    images, nsfw, seeds = process_prompt(prompt, guidance, iterations, height, width, batch_size, seed, seed_step)
    
    grid = image_grid(images,1,batch_size)
    with io.BytesIO() as grid_with_data:
        md = PngInfo()
        md.add_text("SD:prompt_id",prompt_id)
        md.add_text("SD:seeds",seeds)
        grid.save(grid_with_data, pnginfo=md)
        return send_file(grid_with_data, mimetype='image/png',download_name=prompt_id+".png")
    
def process_prompt(prompt='', guidance=7.5, iterations=50, height=512, width=512, batch_size=1, seed='', seed_step=0):
    height = min(max_height, height)
    width = min(max_width, width)
    if prompt == '':
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
    
def root_dir():
    return os.path.abspath(os.path.dirname(__file__))
    
def main(argv):
    sdrequests.secret = "stablediffusion"
    ip = "127.0.0.1"
    p = "5555"
    h = 512
    w = 512
    
    try:
        opts, args = getopt.getopt(argv,"",["orchurl","secret","ip","port","lowermem","maxheight","maxwidth","modelpath"])
        for opt, arg in opts:
            if opt == '--orchurl':
                orch_uri = arg
            elif opt == '--secret':
                print("secret set")
                sdrequests.secret = arg
            elif opt == "--ip":
                print("ip set")
                ip = arg
            elif opt == "--port":
                print("port set")
                p = arg
            elif opt == "--maxheight":
                print("max height set")
                h = arg
            elif opt == "--maxwidth":
                print("max width set")
                w = arg
            elif opt == "--modelpath":
                model_path = arg
            elif opt == "--lowermem":
                lower_mem = True
        app.run(host=ip, port=p, ssl_context="adhoc", threaded=True)
        
    except getopt.GetoptError:
        print("error reading options")
        

if __name__=="__main__":
    main(sys.argv[1:])
    
