import torch
from diffusers import StableDiffusionInpaintPipeline
from segment_anything import SamPredictor, sam_model_registry
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

class ModelLoader:
    @staticmethod
    def load_pipeline():
        pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float32,
        )
        return pipeline

    @staticmethod
    def load_sam(model_path, device):
        sam = sam_model_registry['vit_h'](checkpoint=model_path)
        sam.to(device)
        return SamPredictor(sam)

    @staticmethod
    def load_dino(model_id, device):
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        return processor, model