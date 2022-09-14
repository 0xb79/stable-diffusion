from flask import Flask, request, send_file, make_response, jsonify
import sys, getopt, os, threading, hashlib, torch
from PIL.PngImagePlugin import PngInfo
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
app = Flask(__name__)

access_token = "enter access token"
model path = ""
use8g = False
max_height = 512
max_width = 512
max_sessions = 2

sdrequests = Sdrequests()
t = threading.BoundedSemaphore(max_sessions)

# this will substitute the default PNDM scheduler for K-LMS  
lms = LMSDiscreteScheduler(
    beta_start=0.00085, 
    beta_end=0.012, 
    beta_schedule="scaled_linear"
)

pipe = None
if use8g == False:
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", 
        scheduler=lms,
        use_auth_token=access_token
    )
else:
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", 
        revision="fp16", 
        torch_dtype=torch.float16,
        scheduler=lms,
        use_auth_token=access_token
    )

if torch.cuda.is_available():
    print("using cuda")
    pipe = pipe.to("cuda:1")
    if use8g:
        pipe.enable_attention_slicing()

@app.route("/setaccesstoken", methods=['POST'])
def set_access_token():
    token = request.values.get("token")
    if token[:2] == "hf" and len(token) == 37:
        access_token = token
        return make_response("Ok", 200)
    else:
        return make_response("token not input correctly (start with 'hf' and be 37 characters", 400)

@app.route("/setmaxsessions", methods=['POST'])
def set_max_sessions():
    sessions = request.values.get("sessions")
    if sessions.isnumeric():
        t = threading.BoundedSemaphore(int(sessions))
        return make_response("Ok", 200)
    else:
        return make_response("sessions not input as integer", 400)

@app.route("/setgpu", methods=['POST'])
def set_gpu():
    gpu = request.values.get("gpu")
    if gpu.isnumeric():
        pipe = pipe.to("cuda:"+gpu)
        return make_response("Ok", 200)
    else:
        return make_response("gpu not input as integer", 400)

@app.route("/setmaximagesize", methods=['POST'])
def set_max_image_size():
    h = request.values.get("maxheight")
    w = request.values.get("maxwidth")
    
    if h.isnumeric() and w.isnumeric():
        max_height = h
        max_width = w
        return make_response("max image size set", 200)
    else:
        return make_response("must input numeric height and width", 400)

@app.route("/transcoderconfig")
def transcoder_config():
    data = {"maxheight": max_height, "maxwidth": max_width, "max_sessions": max_sessions}
    return jsonify(data)

@app.route("/stable_diffusion", methods=['GET'])
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
    seed = prompt.values.get("seed")
    seed_step = prompt.values.get("seed_step")
    prompt_id = request.headers.get("prompt_id")
    
    #pipe returns [images] and if [nsfw_content_detected] 
    images, nsfw, seeds = process_prompt(prompt, guidance, iterations, height, width, batch_size, seed, seed_step)
    
    
    #prompt_path = os.path.join(root_dir(), "data", prompt + ".png")
    #image.save(prompt_path)
    for i in range(len(images)):
        return send_file(images[i].tobytes(), mimetype='image/png',as_attachment=True,download_name=prompt_id+"|"+i+".png")
    return 

def process_prompt(prompt='', guidance=7.5, iterations=50, height=512, width=512, batch_size=1, seed='', seed_step=0):
    height = min(max_height, height)
    width = min(max_width, width)
    if prompt = '':
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
        
    if batch_size > 1:
        for _ in range(batch_size-1):
            
            if batch_step = 0:
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
    
def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    
def root_dir():
    return os.path.abspath(os.path.dirname(__file__))

def main(argv):
    secret = "stablediffusion"
    ip = "127.0.0.1"
    p = "5555"
    h = 512
    w = 512
    
    try:
      opts, args = getopt.getopt(argv,"",["secret","ip","port","use8g","maxheight","maxwidth"])
    except getopt.GetoptError:
        print("error reading options")
    for opt, arg in opts:
        if opt == '--secret':
            print("secret set")
            sqreqeusts.secret = arg
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
     
     app.run(host=ip, port=p, ssl_context="adhoc", threaded=True)

if __name__=="__main__":
    main(sys.argv[1:])
    
