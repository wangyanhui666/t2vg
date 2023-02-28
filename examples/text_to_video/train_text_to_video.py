import argparse
import hashlib
import itertools
import math
import os
from pathlib import Path
from typing import Optional
import numpy as np

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from src.diffusers import UNetPseudo3DConditionModel
from diffusers.optimization import get_scheduler
from huggingface_hub import HfFolder, Repository, whoami
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from einops import rearrange, repeat
from imageio import mimread

import webdataset as wds
from io import BytesIO

from random import randrange

logger = get_logger(__name__)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--training_urls",
        type=str,
        default="/workdir/datasets/windows_storage/mgif_tiny_f8.tar",
        required=True,
        help="webdataset",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=True,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If not have enough images, additional images will be"
            " sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--dataset_size",
        type=int,
        default=4,
        help=(
            "how many data in your dataset"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution"
    )
    parser.add_argument("--train_text_encoder", action="store_true", help="Whether to train the text encoder")
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--frames_length", type=int, default=8, help="how many frames to fetch"
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--save_every_n_step",
        type=int,
        default=None,
        help="save often as n",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else:
        if args.class_data_dir is not None:
            logger.warning("You need not use --class_data_dir without --with_prior_preservation.")
        if args.class_prompt is not None:
            logger.warning("You need not use --class_prompt without --with_prior_preservation.")

    return args

def rgba2rgb( rgba, background=(255,255,255) ):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros( (row, col, 3), dtype='float32' )
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

    a = np.asarray( a, dtype='float32' ) / 255.0

    R, G, B = background

    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B

    return np.asarray( rgb, dtype='uint8' )

class VideoLatentWebDataset(Dataset):
    def __init__(self, batch_size, training_urls=None, val_urls=None, test_urls=None,
            frames_length=8, num_workers=None):
        super().__init__()
        self.batch_size = batch_size
        self.training_urls = training_urls
        self.val_urls = val_urls
        self.test_urls = test_urls
        #self.num_workers = num_workers if num_workers is not None else batch_size*2
        self.num_workers = num_workers if num_workers is not None else 8
        self.frames_length = frames_length
        if training_urls is not None:
            self.train_dataloader = self._train_dataloader
        if val_urls is not None:
            self.val_dataloader = self._val_dataloader
        if test_urls is not None:
            self.test_dataloader = self._test_dataloader

    def preprocess(self, sample):
        (npz, key) = sample
        example = {}
        f8_list = []
        txt_embed_list = []
        for npz_i in npz:
            np_obj = np.load(BytesIO(npz_i))
            f8_vid = torch.from_numpy(np_obj.f.f8)
            txt_embed = torch.from_numpy(np_obj.f.txt_embed)
            num_frames = f8_vid.shape[0]
            if num_frames > self.frames_length:
                frame_start = randrange(num_frames - self.frames_length)
                f8_list.append(f8_vid[frame_start:frame_start + self.frames_length,:,:,:])
            else:
                f8_list.append(f8_vid)
            #f8_list.append(f8_vid[:self.frames_length,:,:,:])
            txt_embed_list.append(txt_embed)
            np_obj.close()
        f8_batch = torch.stack(f8_list, dim=0)
        txt_embed_batch = torch.stack(txt_embed_list, dim=0)
        f8_batch = rearrange(f8_batch, 'b f c h w -> b c f h w')
        example['f8'] = f8_batch
        example['txt_embed'] = txt_embed_batch
        return example

    def make_loader(self, urls, mode="train"):
        shuffle = 0
        dataset = (
            wds.WebDataset(urls)
            .shuffle(shuffle)
            .to_tuple("npz", "__key__")
        )
        dataset = dataset.batched(self.batch_size, partial=False)
        dataset = dataset.map(self.preprocess)

        loader = wds.WebLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.num_workers,
        )
        loader.batch_size = self.batch_size
        return loader

    def _train_dataloader(self):
        return self.make_loader(self.training_urls, mode="train")

    def _val_dataloader(self):
        return self.make_loader(self.val_urls, mode="val")

    def _test_dataloader(self):
        return self.make_loader(self.test_urls, mode="test")
    
class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        logging_dir=logging_dir,
    )

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    if args.seed is not None:
        set_seed(args.seed)

    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < args.num_class_images:
            torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                safety_checker=None,
                revision=args.revision,
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = args.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)

            sample_dataloader = accelerator.prepare(sample_dataloader)
            pipeline.to(accelerator.device)

            for example in tqdm(
                sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
            ):
                images = pipeline(example["prompt"]).images

                for i, image in enumerate(images):
                    hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.tokenizer_name,
            revision=args.revision,
        )
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
        )

    # Load models and create wrapper for stable diffusion
    #text_encoder = CLIPTextModel.from_pretrained(
    #    args.pretrained_model_name_or_path,
    #    subfolder="text_encoder",
    #    revision=args.revision,
    #)
    #vae = AutoencoderKL.from_pretrained(
    #    args.pretrained_model_name_or_path,
    #    subfolder="vae",
    #    revision=args.revision,
    #)
    #unet = UNet2DConditionModel.from_pretrained(
    #    args.pretrained_model_name_or_path,
    #    subfolder="unet",
    #    revision=args.revision,
    #)
    unet = UNetPseudo3DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
    )

    unet.train()
    #unet.set_use_memory_efficient_attention_xformers(True)
    #unet.disable_gradient_checkpointing()
    # unet.requires_grad_(False)
    # for name, param in unet.named_parameters():
    #     if 'temporal_conv' in name:
    #         param.requires_grad_(True)
    #         print(name)
    # for block in unet.down_blocks:
    #     if hasattr(block, "attentions") and block.attentions is not None:
    #         for attn_block in block.attentions:
    #             for transformer_block in attn_block.transformer_blocks:
    #                 transformer_block.requires_grad_(False)
    #                 transformer_block.attn_temporal.requires_grad_(True)
    #                 transformer_block.norm_temporal.requires_grad_(True)
    # for block in [unet.mid_block,]:
    #     if hasattr(block, "attentions") and block.attentions is not None:
    #         for attn_block in block.attentions:
    #             for transformer_block in attn_block.transformer_blocks:
    #                 transformer_block.requires_grad_(False)
    #                 transformer_block.attn_temporal.requires_grad_(True)
    #                 transformer_block.norm_temporal.requires_grad_(True)
    # for block in unet.up_blocks:
    #     if hasattr(block, "attentions") and block.attentions is not None:
    #         for attn_block in block.attentions:
    #             for transformer_block in attn_block.transformer_blocks:
    #                 transformer_block.requires_grad_(False)
    #                 transformer_block.attn_temporal.requires_grad_(True)
    #                 transformer_block.norm_temporal.requires_grad_(True)

    #unet.set_use_memory_efficient_attention_xformers(True)
    #vae.requires_grad_(False)
    #if not args.train_text_encoder:
    #    text_encoder.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        #if args.train_text_encoder:
        #    text_encoder.gradient_checkpointing_enable()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimize = (
        filter(lambda p: p.requires_grad, unet.parameters()) 
    )
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    noise_scheduler = DDPMScheduler.from_config(args.pretrained_model_name_or_path, subfolder="scheduler")

    #train_dataset = GifVideoDataset(
    #    instance_data_root=args.instance_data_dir,
    #    instance_prompt=args.instance_prompt,
    #    class_data_root=args.class_data_dir if args.with_prior_preservation else None,
    #    class_prompt=args.class_prompt,
    #    tokenizer=tokenizer,
    #    size=args.resolution,
    #    center_crop=args.center_crop,
    #)
    train_dataset = VideoLatentWebDataset(
        batch_size = args.train_batch_size,
        training_urls = args.training_urls,
        frames_length = args.frames_length,
        num_workers = 1
    )

    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]

        # Concat class and instance examples for prior preservation.
        # We do this to avoid doing two forward passes.
        if args.with_prior_preservation:
            input_ids += [example["class_prompt_ids"] for example in examples]
            pixel_values += [example["class_images"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = tokenizer.pad(
            {"input_ids": input_ids},
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }
        return batch

    #train_dataloader = torch.utils.data.DataLoader(
    #    train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, num_workers=1
    #)
    train_dataloader = train_dataset._train_dataloader()

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(args.dataset_size / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    if args.train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    #vae.to(accelerator.device, dtype=weight_dtype)
    #if not args.train_text_encoder:
    #    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(args.dataset_size / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {args.dataset_size}")
    logger.info(f"  Num batches each epoch = {args.dataset_size//args.train_batch_size}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0
    saved_global_step = 0

    unet.train()
    #unet.set_use_memory_efficient_attention_xformers(True)
    #unet.disable_gradient_checkpointing()
    unet.requires_grad_(False)
    for name, param in unet.named_parameters():
        if 'temporal_conv' in name:
            param.requires_grad_(True)
            print(name)
    for block in unet.down_blocks:
        if hasattr(block, "attentions") and block.attentions is not None:
            for attn_block in block.attentions:
                for transformer_block in attn_block.transformer_blocks:
                    transformer_block.requires_grad_(False)
                    transformer_block.attn_temporal.requires_grad_(True)
                    transformer_block.norm_temporal.requires_grad_(True)
    for block in [unet.mid_block,]:
        if hasattr(block, "attentions") and block.attentions is not None:
            for attn_block in block.attentions:
                for transformer_block in attn_block.transformer_blocks:
                    transformer_block.requires_grad_(False)
                    transformer_block.attn_temporal.requires_grad_(True)
                    transformer_block.norm_temporal.requires_grad_(True)
    for block in unet.up_blocks:
        if hasattr(block, "attentions") and block.attentions is not None:
            for attn_block in block.attentions:
                for transformer_block in attn_block.transformer_blocks:
                    transformer_block.requires_grad_(False)
                    transformer_block.attn_temporal.requires_grad_(True)
                    transformer_block.norm_temporal.requires_grad_(True)

    for epoch in range(args.num_train_epochs):
        if args.train_text_encoder:
            text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                #latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = batch["f8"]
                latents = latents * 0.18215
                hint_latent = latents[:,:,:1,:,:]
                input_latents = latents[:,:,1:,:,:]
                #latents = latents.float16()
                #hint_latent = hint_latent.half()
                #input_latents = input_latents.half()
                #latents = latents.to(accelerator.device)
                hint_latent = hint_latent.to(accelerator.device)
                input_latents = input_latents.to(accelerator.device)
                #latents = repeat(latents, 'b c h w -> b c f h w', f=8)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(input_latents)
                bsz = input_latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=input_latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(input_latents, noise, timesteps)

                # Get the text embedding for conditioning
                #encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                encoder_hidden_states = batch['txt_embed'].half().to(accelerator.device)

                mask = torch.zeros([noisy_latents.shape[0], 1, noisy_latents.shape[2], noisy_latents.shape[3], noisy_latents.shape[4]]).to(accelerator.device)
                latent_model_input = torch.cat([noisy_latents, mask, hint_latent.expand(-1,-1,noisy_latents.shape[2],-1,-1)], dim=1).to(accelerator.device)
                # Predict the noise residual
                noise_pred = unet(latent_model_input, timesteps, encoder_hidden_states).sample

                if args.with_prior_preservation:
                    # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
                    noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
                    noise, noise_prior = torch.chunk(noise, 2, dim=0)

                    # Compute instance loss
                    loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="none").mean([1, 2, 3, 4]).mean()

                    # Compute prior loss
                    prior_loss = F.mse_loss(noise_pred_prior.float(), noise_prior.float(), reduction="mean")

                    # Add the prior loss to the instance loss.
                    loss = loss + args.prior_loss_weight * prior_loss
                else:
                    loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        unet.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            # Create the pipeline using using the trained modules and save it.
            if (global_step % args.save_every_n_step) == 0 and (saved_global_step != global_step) and accelerator.is_main_process:
                print("saving at step: " + str(global_step))
                pipeline = StableDiffusionPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    unet=accelerator.unwrap_model(unet),
                    #text_encoder=accelerator.unwrap_model(text_encoder),
                    #text_encoder=None,
                    revision=args.revision,
                )
                pipeline.save_pretrained(args.output_dir + "_" + str(global_step))
                saved_global_step = global_step

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()

        # Create the pipeline using using the trained modules and save it.
        #if accelerator.is_main_process:
        #    pipeline = StableDiffusionPipeline.from_pretrained(
        #        args.pretrained_model_name_or_path,
        #        unet=accelerator.unwrap_model(unet),
        #        #text_encoder=accelerator.unwrap_model(text_encoder),
        #        #text_encoder=None,
        #        revision=args.revision,
        #    )
        #    pipeline.save_pretrained(args.output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
