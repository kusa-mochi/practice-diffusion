import sys
import datetime
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

args = sys.argv
print(args)

model_id = "stabilityai/stable-diffusion-2"

# Use the Euler scheduler here instead
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")

pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler)
pipe = pipe.to("cpu")
prompt = args[1]
image = pipe(prompt).images[0]
# image.save(prompt.replace(' ', '_') + ".png")

# 現在時刻を取得してファイル名にする
now = datetime.datetime.now()
filename = now.strftime("%Y%m%d_%H%M%S") + ".png"
image.save(filename)
print(f"Saved image: {filename}")
