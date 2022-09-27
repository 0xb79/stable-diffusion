from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, LMSDiscreteScheduler
import torch, sys, os, traceback
from PIL import Image
from threading import Lock

class ModelLoadingError(Exception):
    pass

class ProcessingError(Exception):
    pass
    
class StableDiffusionProcessor:
    
    def __init__(self):
        self.lock = Lock()
        self.t2i_pipe = None
        self.i2i_pipe = None
        self.settings = {"accesstoken":"","modelpath":"","slicemem":False,"lowermem":False,"maxheight":512,"maxwidth":512,"device":"cpu","gpu":0,"maxbatchsize":1}
        self.scheduler = None
        pass

    def create_latents(self, seed=[''], batch_size=1, seed_step=0, height=512, width=512):
        generator = torch.Generator(device=self.settings["device"])
        latents = None
        seeds = []
        seed = [] if seed == [''] else seed #set seed to zero length if none provided
        if len(seed) > 0:
            for s in seed:
                generator.manual_seed(int(s))
                seeds.append(int(s))
                
                image_latents = torch.randn(
                                (1, self.t2i_pipe.unet.in_channels, height // 8, width // 8),
                                generator = generator,
                                device = self.settings["device"]
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
                    (1, self.t2i_pipe.unet.in_channels, height // 8, width // 8),
                    generator = generator,
                    device = self.settings["device"]
                )
                latents = image_latents if latents is None else torch.cat((latents, image_latents))
        
        return latents, seeds
        
    def torch_gc(self):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    def process_txt2img_prompt(self, prompt='', guidance=7.5, iterations=50, height=512, width=512, batch_size=1, seed=[''], seed_step=0):
        if self.lock.locked():
            raise ModelLoadingError
            
        height = min(self.settings["maxheight"], int(height))
        width = min(self.settings["maxwidth"], int(width))
        batch_size = min(self.settings["maxbatchsize"], int(batch_size))
        
        output = None
        latents = None
        seeds = None
        try:
            prompt = [prompt] * batch_size
            if torch.cuda.is_available() and "cuda" in self.settings["device"]:
                with torch.autocast("cuda"):
                    latents, seeds = self.create_latents(seed, batch_size, seed_step, height, width)
                    output = self.t2i_pipe(prompt, guidance_scale=guidance, num_inferenece_steps=iterations, height=height, width=width, latents=latents)
            else:
                with torch.autocast("cpu"):
                    latents, seeds = create_latents(seed, batch_size, seed_step, height, width)
                    output = self.t2i_pipe(prompt, guidance_scale=guidance, num_inferenece_steps=iterations, height=height, width=width, latents=latents)
        except RuntimeError as re:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print("stable diffusion processing had error")
            print(exc_type, fname, exc_tb.tb_lineno)
            raise ProcessingError
        except Exception as ee:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(traceback.format_exc())
            raise ProcessingError
        finally:
            self.torch_gc()
        
        if output != None:
            return output.images, seeds
        else:
            return None, None
            

    def process_img2img_prompt(self, prompt, init_img=None, guidance=7.5, strength=.75, iterations=50, batch_size=1, seed=''):
        if self.lock.locked():
            raise ModelLoadingError
        
        if init_img == None:
            return None
        
        generator = torch.Generator(device=self.settings["device"])
        generator.manual_seed(int(s))
        output = None
        
        try:
            prompt = [prompt] * batch_size
            if torch.cuda.is_available() and "cuda" in self.settings["device"]:
                with torch.autocast("cuda"):
                    output = self.i2i_pipe(prompt, init_image=init_img, strength=strength, guidance_scale=guidance, num_inferenece_steps=iterations, generator=generator)
        except RuntimeError as re:
            print("stable diffusion processing had error")
            print(exc_type, fname, exc_tb.tb_lineno)
            raise ProcessingError
        except Exception as ee:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(traceback.format_exc())
            raise ProcessingError
        finally:
            self.torch_gc()
        
        if output != None:
            return output.images
        else:
            return None

    def load_model(self, t2i=True, i2i=True):
        if self.lock.locked():
            raise ModelLoadingError
            
        try:
            self.lock.acquire()
            self.t2i_pipe = None
            self.i2i_pipe = None
            self.scheduler = None
            # this will substitute the default PNDM scheduler for K-LMS  
            self.scheduler = LMSDiscreteScheduler(
                beta_start=0.00085, 
                beta_end=0.012, 
                beta_schedule="scaled_linear"
            )
            
            model = "CompVis/stable-diffusion-v1-4" if self.settings["modelpath"] == "" else self.settings["modelpath"]
            if not self.settings["lowermem"]:
                if t2i:
                    self.t2i_pipe = StableDiffusionPipeline.from_pretrained(
                        model, 
                        scheduler = self.scheduler,
                        use_auth_token = self.settings["accesstoken"]
                    )
                if i2i:
                    self.i2i_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                        model,
                        use_auth_token = self.settings["accesstoken"]
                    )
                    self.i2i_pipe.scheduler = self.scheduler
            else:
                if t2i:
                    self.t2i_pipe = StableDiffusionPipeline.from_pretrained(
                        model, 
                        revision = "fp16", 
                        torch_dtype = torch.float16,
                        scheduler = self.scheduler,
                        use_auth_token = self.settings["accesstoken"]
                    )
                if i2i:
                    self.i2i_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                        model,
                        revision = "fp16", 
                        torch_dtype = torch.float16,
                        use_auth_token = self.settings["accesstoken"]
                    )
                    self.i2i_pipe.scheduler = self.scheduler
            
            if torch.cuda.is_available() and "cuda" in self.settings["device"]:
                print("loading model to cuda gpu "+str(self.settings["gpu"]))
                if t2i:
                    self.t2i_pipe = self.t2i_pipe.to(self.settings["device"])
                if i2i:
                    self.i2i_pipe = self.i2i_pipe.to(self.settings["device"])
                if self.settings["slicemem"]:
                    print("pipe set to use less gpu memory")
                    if t2i:
                        self.t2i_pipe.enable_attention_slicing()
                    if i2i:
                        self.i2i_pipe.enable_attention_slicing()
            else:
                print("loading model to cpu")
                if t2i:
                    self.t2i_pipe = self.t2i_pipe.to("cpu")
                if i2i:
                    self.i2i_pipe = self.i2i_pipe.to("cpu")
        except Exception as e: 
            print(e)
            print("could not load model, make sure accesstoken or modelpath is set")
        finally:
            self.lock.release()
    
    def models_are_loaded(self):
        return self.t2i_pipe != None, self.i2i_pipe != None
        
    