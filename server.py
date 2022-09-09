from flask import Flask, request, send_file, make_response
import sys, getopt, os, threading, hashlib, torch
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
app = Flask(__name__)

access_token = "enter access token"
secret = ""
use8g = False

t = threading.BoundedSemaphore(2)

nsfw_error = "nsfw_error.png binary data"

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

@app.route("/stable_diffusion", methods=['GET'])
def get_result():
    prompt = request.values.get("prompt")
    prompt = prompt.replace(" ","-")
    prompt = prompt.replace(".", "")
    prompt = prompt.replace("/","")
    print(prompt)
    
    return get_image(prompt)

def get_image(prompt):
    try:
        with t:
            if torch.cuda.is_available():
                with torch.autocast("cuda"):
                    image = pipe(prompt).images[0]
            else:
                image = pipe(prompt).images[0]
            
            if image.tobytes() == nsfw_error:
                return make_response("NSFW content detected by model, no image returned", 200)
                
            prompt_path = os.path.join(root_dir(), "data", prompt + ".png")
            image.save(prompt_path)
            torch.cuda.empty_cache()
            
            return send_file(prompt_path, mimetype='image/png')
    except ValueError:
        return make_response("image processing busy, please re-submit", 503)

def root_dir():
    return os.path.abspath(os.path.dirname(__file__))

def is_nsfw(file):
    hash = hashlib.sha1()
    if os.path.isfile(file):
        hash.update(open(file, "rb").read())
    return hash.digest()
    
def main(argv):
    secret = "stablediffusion"
    ip = "127.0.0.1"
    p = "5555"
    
    try:
      opts, args = getopt.getopt(argv,"",["secret","ip","port","use8g"])
    except getopt.GetoptError:
        print("error reading options")
    for opt, arg in opts:
        if opt == '--secret':
            print("secret set")
            secret = arg
        elif opt == "--ip":
            print("ip set")
            ip = arg
        elif opt == "--port":
            print("port set")
            p = arg
     
     app.run(host=ip, port=p, ssl_context="adhoc", threaded=True)

if __name__=="__main__":
    nsfw_error = open(os.path.join(root_dir(), "data", "nsfw-error.png"),"rb").read()
    main(sys.argv[1:])
    
