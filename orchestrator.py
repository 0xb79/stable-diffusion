from flask import Flask, request, send_file, Response, make_response
from torch import autocast, cuda
import torch, os, requests, io
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
app = Flask(__name__)

transcoder_uri = "https://136.32.234.128:5555"

@app.route("/stable_diffusion")
def get_result():
    prompt = request.args.get("prompt")
    if (prompt == None or prompt == ""):
        return make_response("Need to input prompt", 400)
        
    req = request.full_path
    resp = requests.get(transcoder_uri+req, verify=False)
    
    if resp.status_code == 200 :
        print("image received from transcoder")
        return send_file(io.BytesIO(resp.content), mimetype='image/png')
    elif resp.status_code == 503:
        print("transcoder busy")
        return make_response(resp.content, resp.status_code)
    else:
        print("error from transcoder")
        return make_response("could not process prompt", 500)

def root_dir():
    return os.path.abspath(os.path.dirname(__file__))
    
if __name__=="__main__":
    app.run(host="127.0.0.1", port="5555")