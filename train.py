import logging
import math
import os
import shutil
from datetime import timedelta

import accelerate
import datasets
import torch
import torch.nn.functional as F
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from datasets import load_dataset
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from PIL import Image
from dataclasses import dataclass
from dotenv import load_dotenv


import diffusers
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel, DDIMPipeline, DDIMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_tensorboard_available
from diffusers.models import AutoencoderKL


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.21.0.dev0")

logger = get_logger(__name__, log_level="INFO")

@dataclass
class TrainingConfig:
    resolution = 256
    train_batch_size = 16
    eval_batch_size = 16
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_scheduler = "cosine"
    save_model_epochs = 100
    ddpm_num_inference_steps = 1000
    checkpoints_total_limit = 5
    weight_dtype = torch.bfloat16
    checkpointing_steps = 500
    ema_inv_gamma = 1.0
    ema_power = 0.75
    ema_max_decay = 0.999
    

config_all = TrainingConfig()



def train_module(config_all, train_data_dir, output_dir, model_path=None):
    logging_dir = os.path.join(output_dir, "logs")
    accelerator_project_config = ProjectConfiguration(project_dir=output_dir, logging_dir=logging_dir)

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))  # a big number for high resolution or big dataset
    accelerator = Accelerator(
        gradient_accumulation_steps=config_all.gradient_accumulation_steps,
        mixed_precision="no",
        log_with="tensorboard",
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    if not is_tensorboard_available():
        raise ImportError("Make sure to install tensorboard if you want to use it for logging during training.")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                ema_model.save_pretrained(os.path.join(output_dir, "unet_ema"))
                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))
                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()
        def load_model_hook(models, input_dir):
            load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DModel)
            ema_model.load_state_dict(load_model.state_dict())
            ema_model.to(accelerator.device)
            del load_model
            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()
                # load diffusers style into model
                load_model = UNet2DModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)
        
        
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
        
    if model_path is not None:
        model = UNet2DModel.from_pretrained(
        model_path,
        subfolder="unet",
        torch_dtype=torch.float32
    )
    else:
        # Initialize the model
        model = UNet2DModel(
            sample_size=256,
            in_channels=4,
            out_channels=4,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )


    # Create EMA for the model.
    ema_model = EMAModel(
        model.parameters(),
        decay=config_all.ema_max_decay,
        use_ema_warmup=True,
        inv_gamma=config_all.ema_inv_gamma,
        power=config_all.ema_power,
        model_cls=UNet2DModel,
        model_config=model.config,
    )

    vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", from_tf=True)



    model.to('cuda', dtype=config_all.weight_dtype)
    vae.to('cuda', dtype=config_all.weight_dtype)
    ema_model.to('cuda', dtype=config_all.weight_dtype)

    # Initialize the scheduler
    if model_path is not None:
        noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder='scheduler')
    else: 
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
            prediction_type="epsilon",
        )

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config_all.learning_rate,
        betas=(0.95, 0.99),
        weight_decay=1e-6,
        eps=1e-08,
    )

    # Load the dataset
    train_dataset = load_dataset(
                train_data_dir,
                split='train'
    )

    augmentations = transforms.Compose(
            [
                transforms.Resize(config_all.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomCrop(config_all.resolution),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
    )

    def transform_images(examples):
        images = [augmentations(image.convert("RGB")) for image in examples["image"]]
        return {"input": images}

    train_dataset.set_transform(transform_images)
    train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config_all.train_batch_size, shuffle=True
    )

    # Initialize the learning rate scheduler
    lr_scheduler = get_scheduler(
        config_all.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=500 * config_all.gradient_accumulation_steps,
        num_training_steps=(len(train_dataloader) * config_all.num_epochs),
    )

    # Prepare everything with our `accelerator`.
    model, vae, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, vae, optimizer, train_dataloader, lr_scheduler
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if model_path is not None:
        if accelerator.is_main_process:
            accelerator.init_trackers("dreambooth")
    else:
        if accelerator.is_main_process:
            run = os.path.split(__file__)[-1].split(".")[0]
            accelerator.init_trackers(run)
            
    total_batch_size = config_all.train_batch_size * accelerator.num_processes * config_all.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config_all.gradient_accumulation_steps)
    max_train_steps = config_all.num_epochs * num_update_steps_per_epoch

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {config_all.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {config_all.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config_all.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    first_epoch = 0

       
    # Train!
    for epoch in range(first_epoch, config_all.num_epochs):
        model.train()
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        for batch in (train_dataloader):
            

            with accelerator.accumulate(model):

                # Convert images to latent space
                latents = vae.encode(batch["input"].to(dtype=config_all.weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device='cuda')
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Predict the noise residual
                model_pred = model(noisy_latents, timesteps).sample


                # Compute loss
                loss = F.mse_loss(model_pred.float(), noise.float())
                loss = loss.mean()
                
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                ema_model.step(model.parameters())
                progress_bar.update(1)
                global_step += 1

                if global_step % config_all.checkpointing_steps == 0:
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    if config_all.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        if len(checkpoints) >= config_all.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - config_all.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)

                    if accelerator.is_main_process:
                        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            logs["ema_decay"] = ema_model.cur_decay_value
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
        progress_bar.close()

    accelerator.wait_for_everyone()

    # Generate sample images for visual inspection
    if accelerator.is_main_process:
        if epoch % config_all.save_model_epochs == 0 or epoch == config_all.num_epochs - 1:
            # save the model
            unet = accelerator.unwrap_model(model)
            unet.to('cuda')

            ema_model.store(unet.parameters())
            ema_model.copy_to(unet.parameters())
            
            pipeline = DDPMPipeline(
                unet=unet,
                scheduler=noise_scheduler,
            )

            pipeline.to('cuda')

            pipeline.save_pretrained(output_dir)

            ema_model.restore(unet.parameters())

    accelerator.end_training()
    
    
    
def sample_image(config_all, model_path, train_or_finetuned):
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
        batch_size=config_all.eval_batch_size,
        num_inference_steps=40,
    ).images
    
    for i, image in enumerate(images):
        image.save(f"Sample_image_{i}_{train_or_finetuned}.png", format='PNG')


def finetune():
    pass


if __name__ == "__main__":
    #Load .env file
    load_dotenv()
    
    train_data_dir = os.getenv("TRAIN_DATABASE")
    finetune_data_dir = os.getenv("FINETUNE_DATABASE")
    output_dir = os.getenv("OUTPUT_DIR")
    output_finetune_dir = os.getenv("OUTPUT_DIR_FINETUNE")
    
    
    torch.cuda.empty_cache()
    train_module(config_all=config_all, train_data_dir=train_data_dir, output_dir=output_dir)
    torch.cuda.empty_cache()
    sample_image(config_all=config_all, model_path=output_dir, train_or_finetuned="train")
    torch.cuda.empty_cache()
    train_module(config_all=config_all, train_data_dir=finetune_data_dir, output_dir=output_finetune_dir, model_path="output_dir")
    torch.cuda.empty_cache()
    sample_image(config_all=config_all, model_path=output_finetune_dir, train_or_finetuned="finetuned")
