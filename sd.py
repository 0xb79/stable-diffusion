from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, LMSDiscreteScheduler
import torch, sys, os, traceback, logging
from PIL import Image
from threading import Lock

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s',level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelLoadingError(Exception):
    pass

class ProcessingError(Exception):
    pass
    
class StableDiffusionProcessor:
    def __init__(self):
        self.lock = Lock()
        self.t2i_pipe = None
        self.i2i_pipe = None
        self.settings = {"accesstoken":"","modelpath":"","slicemem":True,"lowermem":True,"maxheight":512,"maxwidth":512,"device":"cpu","gpu":0,"maxbatchsize":1}
        self.scheduler = None
        pass

    def create_latents(self, seed=[''], batchsize=1, seedstep=0, height=512, width=512):
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
        
        if batchsize > len(seed):
            addl = batchsize - len(seed)
            last_seed = 0 if len(seed) == 0 else seed[-1]
            if addl > 0:
                for _ in range(addl):
                    if seedstep == 0 or seed == []:
                        # Get a new random seed, store it and use it as the generator state
                        s = generator.seed()
                        seeds.append(s)
                    else:
                        #update the seed by the step
                        new_seed = last_seed+int(seedstep)
                        s = generator.manual_seed(new_seed)
                        seeds.append(s)
                        last_seed = new_seed
                    
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
    
    def txt2img_processing(self, prompt, prompt_id, guidance, iterations, height, width, batchsize, seed, seedstep):
        output = None
        latents = None
        seeds = None
        error = None
        
        try:
            prompt_list = [prompt] * batchsize
            if torch.cuda.is_available() and "cuda" in self.settings["device"]:
                with torch.autocast("cuda"):
                    latents, seeds = self.create_latents(seed, batchsize, seedstep, height, width)
                    output = self.t2i_pipe(prompt_list, guidance_scale=guidance, num_inference_steps=iterations, height=height, width=width, latents=latents)
            else:
                with torch.autocast("cpu"):
                    latents, seeds = self.create_latents(seed, batchsize, seedstep, height, width)
                    output = self.t2i_pipe(prompt_list, guidance_scale=guidance, num_inference_steps=iterations, height=height, width=width, latents=latents)
        except RuntimeError as re:
            error = "processing failed, too much memory used"
            logger.error(error, exc_info=True)
        except Exception as ee:
            error = "processing failed"
            logger.error(error, exc_info=True)
        finally:
            self.torch_gc()
        
        if output != None:
            return output.images, seeds, prompt_list, prompt_id, error
        else:
            return None, None, prompt_list, prompt_id, error
        
    def img2img_processing(self, prompt, prompt_id, init_img, guidance, strength, iterations, batchsize, seed):
        output = None
        error = ""
        try:
            prompt = [prompt] * i_batchsize
            if torch.cuda.is_available() and "cuda" in self.settings["device"]:
                with torch.autocast("cuda"):
                    output = self.i2i_pipe(prompt, init_image=init_img, strength=strength, guidance_scale=guidance, num_inference_steps=iterations, generator=generator)
        except RuntimeError as re:
            error = "processing failed, too much memory used"
            logger.error(error, exc_info=True)
        except Exception as ee:
            error = "processing failed"
            logger.error(error,exc_info=True)
        finally:
            self.torch_gc()
        
        if output != None:
            return output.images, seed, prompt, prompt_id, error
        else:
            return None, None, prompt, prompt_id, error
        
    def process_txt2img_prompt(self, prompt='', prompt_id='', guidance=7.5, iterations=50, height=512, width=512, batchsize=1, seed=[''], seedstep=0):
        if self.lock.locked():
            raise ModelLoadingError
            
        proc_height = min(self.settings["maxheight"], int(height))
        proc_width = min(self.settings["maxwidth"], int(width))
        proc_batchsize = min(self.settings["maxbatchsize"], int(batchsize))
        completed = 0
        seeds = []
        images = []
        prompts = []
        while completed < batchsize:
            pb = min(proc_batchsize, batchsize-completed)
            logger.info("processing "+str(completed+pb)+" of "+str(batchsize))
            proc_images, proc_seeds, prompt_list, prompt_id, error = self.txt2img_processing(prompt, prompt_id, guidance, iterations, proc_height, proc_width, pb, seed, seedstep)
            if error == None:
                prompts.extend(prompt_list)
                images.extend(proc_images)
                seeds.extend(proc_seeds)
            completed += pb
        
        return images, seeds, prompts, prompt_id, error
        
    def process_img2img_prompt(self, prompt, prompt_id='', init_img=None, guidance=7.5, strength=.75, iterations=50, batchsize=1, seed=''):
        if self.lock.locked():
            raise ModelLoadingError
        
        if init_img == None:
            return None
        
        generator = torch.Generator(device=self.settings["device"])
        if seed != '':
            generator.manual_seed(int(seed))
        else:
            seed = generator.seed()
        
        proc_batchsize = min(self.settings["maxbatchsize"], int(batchsize))
        completed = 0
        seeds = []
        images = []
        prompts = []
        while completed < batchsize:
            pb = min(proc_batchsize, batchsize-completed)
            logger.info("processing "+str(completed+pb)+" of "+str(batchsize))
            proc_images, proc_seeds, prompt_list, prompt_id, error = self.img2img_processing(prompt, prompt_id, init_img, guidance, strength, iterations, pb, seed)
            if error == None:
                prompts.extend(prompt_list)
                images.extend(proc_images)
                seeds.extend(proc_seeds)
            completed += pb
            
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
                    print("loading txt2img")
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
        except Exception as ee:
            print(ee)
            print("model not loaded, error on loading")
        finally:
            self.lock.release()
    
    def models_are_loaded(self):
        return self.t2i_pipe != None, self.i2i_pipe != None
        
    