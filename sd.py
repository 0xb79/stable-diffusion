from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
import torch, sys, os, traceback


pipe = None
scheduler = None
settings = {"accesstoken":"","modelpath":"","lowermem":False,"maxheight":512,"maxwidth":512,"device":"cpu","gpu":0,"maxbatchsize":32}

def process_txt2img_prompt(prompt='', guidance=7.5, iterations=50, height=512, width=512, batch_size=1, seed=[''], seed_step=0):
    global pipe
    global settings
    height = min(settings["maxheight"], int(height))
    width = min(settings["maxwidth"], int(width))
    batch_size = min(settings["maxbatchsize"], int(batch_size))
    
    try:
        prompt = [prompt] * batch_size
        if torch.cuda.is_available() and "cuda" in settings["device"]:
            with torch.autocast("cuda"):
                latents, seeds = create_latents(seed, batch_size, seed_step, height, width)
                output = pipe(prompt, guidance=guidance, iterations=iterations, height=height, width=width, latents=latents)
                return output.images, output.nsfw_content_detected, seeds, False
            torch_gc()
        else:
            with torch.autocast("cpu"):
                latents, seeds = create_latents(seed, batch_size, seed_step, height, width)
                output = pipe(prompt, guidance=guidance, iterations=iterations, height=height, width=width, latents=latents)
                return output.images, output.nsfw_content_detected, seeds, False
            
        
    except Exception as ee:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(traceback.format_exc())
        return None, None, None, False

def process_img2img(prompt, img, guidance=7.5, iterations=50, height=512, width=512, batch_size=1, seed=[''], seed_step=0):
    return None
    
def create_latents(seed=[''], batch_size=1, seed_step=0, height=512, width=512):
    global pipe
    global settings
    generator = torch.Generator(device=settings["device"])
    latents = None
    seeds = []
    seed = [] if seed == [''] else seed #set seed to zero length if none provided
    
    if len(seed) > 0 and seed != ['']:
        for s in seed:
            generator.manual_seed(int(s))
            seeds.append(int(s))
        else:
            s = generator.seed()
            seeds.append(s)
            image_latents = torch.randn(
                (1, pipe.unet.in_channels, height // 8, width // 8),
                generator = generator,
                device = settings["device"]
            )
            latents = image_latents if latents is None else torch.cat((latents, image_latents))
    
    if batch_size > len(seed):
        addl = batch_size - len(seed)
        for _ in range(addl):
            if seed_step == 0 or seed == []:
                # Get a new random seed, store it and use it as the generator state
                s = generator.seed()
                seeds.append(s)
            else:
                #update the seed by the step
                s = generator.manual_seed(int(seed[-1])+int(seed_step))
                seeds.append(s)
            
            image_latents = torch.randn(
                (1, pipe.unet.in_channels, height // 8, width // 8),
                generator = generator,
                device = settings["device"]
            )
            latents = image_latents if latents is None else torch.cat((latents, image_latents))
    
    return latents, seeds
    
def load_model():
    global pipe
    global scheduler
    global settings
    
    try:
        pipe = None
        scheduler = None
        # this will substitute the default PNDM scheduler for K-LMS  
        scheduler = LMSDiscreteScheduler(
            beta_start=0.00085, 
            beta_end=0.012, 
            beta_schedule="scaled_linear"
        )
        
        model = "CompVis/stable-diffusion-v1-4" if settings["modelpath"] == "" else settings["modelpath"]
        if settings["lowermem"] == False:
            pipe = StableDiffusionPipeline.from_pretrained(
                model, 
                scheduler=scheduler,
                use_auth_token=settings["accesstoken"]
            )
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                model, 
                revision="fp16", 
                torch_dtype=torch.float16,
                scheduler=scheduler,
                use_auth_token=settings["accesstoken"]
            )
        
        if torch.cuda.is_available() and "cuda" in settings["device"]:
            print("loading model to cuda gpu "+str(settings["gpu"]))
            pipe = pipe.to(settings["device"])
            if settings["lowermem"]:
                print("pipe set to use less gpu memory")
                pipe.enable_attention_slicing()
        else:
            print("loading model to cpu")
            pipe = pipe.to("cpu")
        
    except Exception as e: 
        print(e)
        print("could not load model, make sure accesstoken or modelpath is set")
    
def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()