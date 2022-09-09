from flask import Flask, request, send_file, make_response
from torch import autocast, cuda
import torch, os, threading
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
app = Flask(__name__)

access_token = "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"

t = threading.BoundedSemaphore(2)

nsfw_error = "nsfw_error.png binary data"

# this will substitute the default PNDM scheduler for K-LMS  
lms = LMSDiscreteScheduler(
    beta_start=0.00085, 
    beta_end=0.012, 
    beta_schedule="scaled_linear"
)

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", 
    scheduler=lms,
    use_auth_token=access_token
)

if cuda.is_available():
    print("using cuda")
    pipe = pipe.to("cuda:1")

@app.route("/setaccesstoken")
def set_access_token():
    token = request.args.get("token")
    if token[:2] == "hf" and len(token) == 37:
        access_token = token
        return make_response("Ok", 200)
    else:
        return make_response("token not input correctly (start with 'hf' and be 37 characters", 400)

@app.route("/setmaxsessions")
def set_max_sessions():
    sessions = request.args.get("sessions")
    if sessions.isnumeric():
        t = threading.BoundedSemaphore(int(sessions))
        return make_response("Ok", 200)
    else:
        return make_response("sessions not input as integer", 400)

@app.route("/setgpu")
def set_gpu():
    gpu = request.args.get("gpu")
    if gpu.isnumeric():
        pipe = pipe.to("cuda:"+gpu)
        return make_response("Ok", 200)
    else:
        return make_response("gpu not input as integer", 400)

@app.route("/stable_diffusion")
def get_result():
    prompt = request.args.get("prompt")
    prompt = prompt.replace(" ","-")
    prompt = prompt.replace(".", "")
    prompt = prompt.replace("/","")
    print(prompt)
    
    return get_image(prompt)

def get_image(prompt):
    try:
        with t:
            if cuda.is_available():
                with autocast("cuda"):
                    image = pipe(prompt).images[0] 
            else:
                image = pipe(prompt).images[0]
            
            if image == nsfw_error:
                return make_response("NSFW content detected by model, no image returned", 200)
                
            prompt_path = os.path.join(root_dir(), "data", prompt + ".png")
            image.save(prompt_path)
            torch.cuda.empty_cache()
            
            return send_file(prompt_path, mimetype='image/png')
    except ValueError:
        return make_response("image processing busy, please re-submit", 503)

def root_dir():
    return os.path.abspath(os.path.dirname(__file__))

if __name__=="__main__":
    nsfw_error = open(os.path.join(root_dir(), "data", "nsfw-error.png"),"rb").read()
    app.run(host="0.0.0.0", port="5555", ssl_context="adhoc", threaded=True)
