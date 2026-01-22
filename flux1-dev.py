import torch
from diffusers import FluxPipeline
import time, os
from datetime import datetime

start = time.time()

# Load original FLUX.1-dev model (full precision)
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
)
pipeline.enable_model_cpu_offload()
# Load LoRA
lora_path = "brian_8_4000_000004000.safetensors"
print(f"Loading LoRA from: {lora_path}")

try:
    # Load LoRA weights with adapter name
    pipeline.load_lora_weights(".", weight_name=lora_path, adapter_name="brian_lora")
    print("LoRA loaded successfully")
    
    # Set LoRA scale
    pipeline.set_adapters(["brian_lora"], adapter_weights=[1.0])
    print("LoRA adapter activated")
    
except Exception as e:
    print(f"Error loading LoRA: {e}")
    print("Continuing without LoRA...")

prompt = "A photorealistic waist-up portrait of Brian, Slim build, with Undercut Black hair. He is wearing Charcoal a relaxed t-shirt and jacket. The background is a blurred busy city street with evening bokeh lights. Professional photography, 50mm lens, highly detailed, camera zoomed out to show the full torso, framing cutting off at the belt line, arms down at sides extending out of the bottom of the frame."
print(f"\nStarting inference with prompt: {prompt}")
print("This may take several minutes...")

try:
    images_ = pipeline(
        prompt,
        width=1024,
        height=1024,
        guidance_scale=3.5,
        num_inference_steps=40,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images
    print(f"Inference complete! Generated {len(images_)} image(s)")
except Exception as e:
    print(f"Error during inference: {e}")
    raise

# Save output
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

for idx, image in enumerate(images_):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/brian_original_{timestamp}_{idx}.png"
    image.save(filename)

end = time.time()
print(f"SUCCESS! Total time: {end - start:.2f} seconds")
