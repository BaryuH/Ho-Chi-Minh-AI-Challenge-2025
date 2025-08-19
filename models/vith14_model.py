import torch
import torch.nn.functional as F
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer
import numpy as np

class VITH14Model:
    def __init__(self, device=None):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading OpenCLIP model on {self.device}...")
        self.model, self.preprocess = create_model_from_pretrained(
            'hf-hub:apple/DFN5B-CLIP-ViT-H-14-384'
        )
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = get_tokenizer('ViT-H-14')
        print("OpenCLIP model loaded!")

    def _normalize(self, vec: torch.Tensor) -> torch.Tensor:
        return F.normalize(vec, p=2, dim=-1)

    def get_image_embedding(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            img_emb = self.model.encode_image(img_tensor)  
            img_emb = self._normalize(img_emb)
        return img_emb.cpu().numpy().astype('float32').squeeze(0) 

    def get_text_embedding(self, text: str) -> np.ndarray:
        text_tokens = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            txt_emb = self.model.encode_text(text_tokens)  
            txt_emb = self._normalize(txt_emb)
        return txt_emb.cpu().numpy().astype('float32').squeeze(0)  
