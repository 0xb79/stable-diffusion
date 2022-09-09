from torch import autocast
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
from IPython.display import Image
import os

prompt = "a photo of an astronaut riding a horse on mars"
access_token = "enter access token"

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
).to("cuda")

def get_image():
    file_name = os.path.join(root_dir(), "data", prompt.replace(" ", "-")+".png")
    
    with autocast("cuda"):
        image = pipe(prompt).images[0]  
        image.save(file_name)
    
    torch.cuda.empty_cache()
    
def root_dir():
    return os.path.abspath(os.path.dirname(__file__))

if __name__=="__main__":
    get_image()