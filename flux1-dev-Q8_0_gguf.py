import torch
from diffusers import FluxPipeline, FluxTransformer2DModel, GGUFQuantizationConfig
import time, os
from datetime import datetime

start = time.time()

# Download Q8 GGUF model (best quality GGUF)
gguf_path = "https://huggingface.co/city96/FLUX.1-dev-gguf/blob/main/flux1-dev-Q8_0.gguf"

transformer = FluxTransformer2DModel.from_single_file(
    gguf_path,
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16,
)

pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)
pipeline.enable_model_cpu_offload()
# Load LoRA
lora_path = "brian_8_4000_000004000.safetensors"
pipeline.load_lora_weights(".", weight_name=lora_path, adapter_name="brian_lora")
pipeline.set_adapters(["brian_lora"], adapter_weights=[1.0])

prompt = "A photorealistic waist-up portrait of Brian, Slim build, with Undercut Black hair. He is wearing Charcoal a relaxed t-shirt and jacket. The background is a blurred busy city street with evening bokeh lights. Professional photography, 50mm lens, highly detailed, camera zoomed out to show the full torso, framing cutting off at the belt line, arms down at sides extending out of the bottom of the frame."

images_ = pipeline(
    prompt,
    width=1024,
    height=1024,
    guidance_scale=3.5,
    num_inference_steps=40,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images

output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
images_[0].save(f"{output_dir}/brian_gguf_{timestamp}.png")

print(f"SUCCESS! Total time: {time.time() - start:.2f} seconds")