import numpy as np
import torch
import wandb
from PIL import Image
from accelerate.logging import get_logger


def log_validation(pipeline, args, accelerator, step, logger=None):
    """
    Run validation during training and log images to trackers.

    Args:
        pipeline: The pipeline used for inference
        args: Training arguments
        accelerator: Accelerator instance
        step: Current training step
        logger: Logger instance (defaults to get_logger(__name__) if None)
    """
    if logger is None:
        logger = get_logger(__name__)

    logger.info("Running validation... ")

    pipeline.set_progress_bar_config(disable=True)

    if args.seed is None:
        generator = None
    else:
        generator = torch.manual_seed(args.seed)

    if len(args.validation_image) == len(args.validation_prompt):
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt
    elif len(args.validation_image) == 1:
        validation_images = args.validation_image * len(args.validation_prompt)
        validation_prompts = args.validation_prompt
    elif len(args.validation_prompt) == 1:
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt * len(args.validation_image)
    else:
        raise ValueError(
            "number of `args.validation_image` and `args.validation_prompt` should be checked in `parse_args`"
        )

    image_logs = []

    for validation_prompt, validation_image in zip(validation_prompts, validation_images):
        validation_image = Image.open(validation_image).convert("RGB")
        images = []

        with accelerator.autocast(), torch.inference_mode():
            for _ in range(args.num_validation_images):
                image = pipeline(
                    validation_prompt, control_image=validation_image, num_inference_steps=20, generator=generator
                ).images[0]
                images.append(image)

        image_logs.append(
            {"validation_image": validation_image, "images": images, "validation_prompt": validation_prompt}
        )

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]

                tracker.writer.add_image(
                    "Controlnet conditioning", np.asarray([validation_image]), step, dataformats="NHWC"
                )

                formatted_images = []
                for image in images:
                    formatted_images.append(np.asarray(image))

                formatted_images = np.stack(formatted_images)

                tracker.writer.add_images(validation_prompt, formatted_images, step, dataformats="NHWC")
        elif tracker.name == "wandb":
            formatted_images = []

            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]

                formatted_images.append(wandb.Image(validation_image, caption="Controlnet conditioning"))

                for image in images:
                    image = wandb.Image(image, caption=validation_prompt)
                    formatted_images.append(image)

            tracker.log({"validation": formatted_images})
        else:
            logger.warning(f"image logging not implemented for {tracker.name}")

    return image_logs