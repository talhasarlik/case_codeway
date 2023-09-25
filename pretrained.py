import torch
from diffusers import UNet2DModel, DDIMPipeline, DDIMScheduler
from PIL import Images

def sample_image(model_path, train_or_finetuned):
    config = UNet2DModel.load_config(f"{model_path}/unet")
    unet = UNet2DModel.from_config(config)
    unet.to('cuda')
    scheduler_config = DDIMScheduler.load_config(f"{model_path}/scheduler")
    scheduler = DDIMScheduler.from_config(scheduler_config)
    scheduler.set_timesteps(num_inference_steps=1000)

    pipeline = DDIMPipeline(
        unet=unet,
        scheduler=scheduler,
        
    )
    pipeline.to('cuda')

    generator = torch.Generator(device='cuda').manual_seed(0)
    # run pipeline in inference (sample random noise and denoise)
    images = pipeline(
        generator=generator,
        batch_size=16,
        num_inference_steps=40,
    ).images
    
    for i, image in enumerate(images):
        image.save(f"Sample_image_{i}_{train_or_finetuned}.png", format='PNG')
        

if __name__ == "__main__":
    
    output_dir = 'pretrained_models/output_dir'
    output_finetune_dir = 'pretrained_models/output_finetune_dir'
    # Sample images for normal pretrained model
    sample_image(model_path=output_dir, train_or_finetuned="train")
    torch.cuda.empty_cache()
    
    # Sample images for finetuned pretrained model
    sample_image(model_path=output_finetune_dir, train_or_finetuned="train")
    torch.cuda.empty_cache()