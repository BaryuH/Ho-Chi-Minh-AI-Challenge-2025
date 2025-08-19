import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import XLMRobertaTokenizer
from PIL import Image
import numpy as np
import math
from tqdm import tqdm
from torchvision import transforms
from timm.models.layers import trunc_normal_
from torchscale.model.BEiT3 import BEiT3
from torchscale.architecture.config import EncoderConfig
from typing import List, Tuple, Optional, Union
from pathlib import Path


class ModelConfig:
    @staticmethod
    def get_base_config(
        img_size: int = 224,
        patch_size: int = 16,
        drop_path_rate: float = 0,
        checkpoint_activations: Optional[bool] = None,
        mlp_ratio: int = 4,
        vocab_size: int = 64010,
        **kwargs
    ) -> EncoderConfig:
        return EncoderConfig(
            img_size=img_size,
            patch_size=patch_size,
            vocab_size=vocab_size,
            multiway=True,
            layernorm_embedding=False,
            normalize_output=True,
            no_output_layer=True,
            drop_path_rate=drop_path_rate,
            encoder_embed_dim=768,
            encoder_attention_heads=12,
            encoder_ffn_embed_dim=int(768 * mlp_ratio),
            encoder_layers=12,
            checkpoint_activations=checkpoint_activations
        )

    @staticmethod
    def get_large_config(
        img_size: int = 224,
        patch_size: int = 16,
        drop_path_rate: float = 0,
        checkpoint_activations: Optional[bool] = None,
        mlp_ratio: int = 4,
        vocab_size: int = 64010,
        **kwargs
    ) -> EncoderConfig:
        return EncoderConfig(
            img_size=img_size,
            patch_size=patch_size,
            vocab_size=vocab_size,
            multiway=True,
            layernorm_embedding=False,
            normalize_output=True,
            no_output_layer=True,
            drop_path_rate=drop_path_rate,
            encoder_embed_dim=1024,
            encoder_attention_heads=16,
            encoder_ffn_embed_dim=int(1024 * mlp_ratio),
            encoder_layers=24,
            checkpoint_activations=checkpoint_activations,
        )


class BEiT3Base(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.beit3 = BEiT3(config)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def get_num_layers(self) -> int:
        return self.beit3.encoder.num_layers

    @torch.jit.ignore
    def no_weight_decay(self) -> set:
        return {
            'pos_embed', 'cls_token', 
            'beit3.encoder.embed_positions.A.weight', 
            'beit3.vision_embed.cls_token', 
            'logit_scale'
        }


class BEiT3ForRetrieval(BEiT3Base):    
    def __init__(self, config: EncoderConfig):
        super().__init__(config)
        embed_dim = config.encoder_embed_dim
        
        self.language_head = nn.Linear(embed_dim, embed_dim, bias=False)
        self.vision_head = nn.Linear(embed_dim, embed_dim, bias=False)
        self.language_head.apply(self._init_weights)
        self.vision_head.apply(self._init_weights)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(
        self, 
        image: Optional[torch.Tensor] = None, 
        text_description: Optional[torch.Tensor] = None, 
        padding_mask: Optional[torch.Tensor] = None, 
        only_infer: bool = False,
        **kwargs
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        vision_cls = None
        language_cls = None

        if image is not None:
            outputs = self.beit3(
                textual_tokens=None,
                visual_tokens=image,
                text_padding_position=None,
            )
            x = outputs["encoder_out"]
            vision_cls = self.vision_head(x[:, 0, :])
            vision_cls = F.normalize(vision_cls, dim=-1)

        if text_description is not None:
            outputs = self.beit3(
                textual_tokens=text_description,
                visual_tokens=None,
                text_padding_position=padding_mask,
            )
            x = outputs["encoder_out"]
            language_cls = self.language_head(x[:, 0, :])
            language_cls = F.normalize(language_cls, dim=-1)

        if only_infer:
            return vision_cls, language_cls
        
        return None, vision_cls, language_cls


class ModelFactory:    
    SUPPORTED_MODELS = {
        'beit3_base_patch16_384_retrieval': ('base', BEiT3ForRetrieval),
        'beit3_large_patch16_384_retrieval': ('large', BEiT3ForRetrieval),
    }
    
    @classmethod
    def create_model(
        cls, 
        model_name: str, 
        pretrained: bool = False, 
        **kwargs
    ) -> BEiT3Base:
        if model_name not in cls.SUPPORTED_MODELS:
            raise ValueError(f"Unknown model: {model_name}. Supported models: {list(cls.SUPPORTED_MODELS.keys())}")
        
        config_type, model_class = cls.SUPPORTED_MODELS[model_name]
        
        if config_type == 'base':
            config = ModelConfig.get_base_config(**kwargs)
        else:  # large
            config = ModelConfig.get_large_config(**kwargs)
        
        return model_class(config)


class ImagePreprocessor:
    def __init__(self, image_size: int = 384):
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    def __call__(self, image_path: Union[str, Path]) -> torch.Tensor:
        image = Image.open(image_path).convert('RGB')
        return self.transform(image).unsqueeze(0)


class TextPreprocessor:
    
    def __init__(self, tokenizer: XLMRobertaTokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens_orig = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens_orig)
        
        if len(token_ids) > self.max_length - 2:
            token_ids = token_ids[:self.max_length - 2]

        tokens = [self.tokenizer.bos_token_id] + token_ids + [self.tokenizer.eos_token_id]
        num_tokens = len(tokens)
        
        padding_mask = [0] * num_tokens + [1] * (self.max_length - num_tokens)
        tokens_padded = tokens + [self.tokenizer.pad_token_id] * (self.max_length - num_tokens)

        padding_mask_tensor = torch.tensor(padding_mask).reshape(1, -1)
        token_ids_tensor = torch.tensor(tokens_padded).reshape(1, -1)

        return token_ids_tensor, padding_mask_tensor


class ModelLoader:
    @staticmethod
    def load_model_and_tokenizer(
        model_path: Union[str, Path],
        sentencepiece_model_path: Union[str, Path],
        device: str = 'cpu',
        model_type: str = 'base',
        **kwargs
    ) -> Tuple[BEiT3Base, XLMRobertaTokenizer]:
        print(f"Loading model from: {model_path}")
        print(f"Loading tokenizer from: {sentencepiece_model_path}")
        
        model_name = f'beit3_{model_type}_patch16_384_retrieval'
        
        model = ModelFactory.create_model(model_name, pretrained=False, **kwargs)
        model.to(device)
        
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
                
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys}")
                
            model.eval()
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model checkpoint: {e}")
            raise
        
        try:
            tokenizer = XLMRobertaTokenizer(str(sentencepiece_model_path))
            print("Tokenizer loaded successfully!")
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            raise
        
        return model, tokenizer


class BEiT3FeatureExtractor:
    def __init__(
        self,
        model_path: Union[str, Path],
        sentencepiece_model_path: Union[str, Path],
        device: str = 'cpu',
        model_type: str = 'base',
        image_size: int = 384,
        max_text_length: int = 128,
        **kwargs
    ):
        self.device = device
        self.model_type = model_type
        
        self.model, self.tokenizer = ModelLoader.load_model_and_tokenizer(
            model_path, 
            sentencepiece_model_path, 
            device, 
            model_type,
            **kwargs
        )
        
        self.image_preprocessor = ImagePreprocessor(image_size)
        self.text_preprocessor = TextPreprocessor(self.tokenizer, max_text_length)
        
        if model_type == 'large':
            self.embed_dim = 1024
        else:
            self.embed_dim = 768
    
    def _ensure_float32_1d(self, vector: np.ndarray) -> np.ndarray:
        # Convert to float32
        vector = vector.astype(np.float32)
        
        # Ensure 1D
        if vector.ndim == 2:
            if vector.shape[0] == 1:
                vector = vector.squeeze(0)  # Remove batch dimension
            else:
                raise ValueError(f"Expected batch size 1, got {vector.shape[0]}")
        
        # Validate dimension
        if vector.shape[0] != self.embed_dim:
            raise ValueError(f"Expected embedding dimension {self.embed_dim}, got {vector.shape[0]}")
        
        return vector

    def get_image_embedding(self, image_path: Union[str, Path]) -> np.ndarray:
        with torch.no_grad():
            image_tensor = self.image_preprocessor(image_path).to(self.device)
            vision_cls, _ = self.model(image=image_tensor, only_infer=True)
            
            embedding = vision_cls.cpu().numpy()
            return self._ensure_float32_1d(embedding)

    def embed_images(
        self, 
        image_paths: List[Union[str, Path]], 
        batch_size: int = 16
    ) -> np.ndarray:
        all_features = []
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing images"):
            batch_paths = image_paths[i:i + batch_size]
            batch_tensors = []
            
            for image_path in batch_paths:
                image_tensor = self.image_preprocessor(image_path)
                batch_tensors.append(image_tensor)
            
            batch_input = torch.cat(batch_tensors, dim=0).to(self.device)
            
            with torch.no_grad():
                vision_cls, _ = self.model(image=batch_input, only_infer=True)
                batch_embeddings = vision_cls.cpu().numpy()

                batch_embeddings = batch_embeddings.astype(np.float32)
                all_features.extend(batch_embeddings)
        
        result = np.array(all_features, dtype=np.float32)
        print(f"Generated {len(image_paths)} image embeddings with shape: {result.shape}")
        return result

    def get_text_embedding(self, text: str) -> np.ndarray:

        with torch.no_grad():
            text_ids, padding_mask = self.text_preprocessor(text)
            text_ids = text_ids.to(self.device)
            padding_mask = padding_mask.to(self.device)
            
            _, language_cls = self.model(
                text_description=text_ids, 
                padding_mask=padding_mask, 
                only_infer=True
            )
            
            embedding = language_cls.cpu().numpy()
            return self._ensure_float32_1d(embedding)

    def embed_texts(
        self, 
        texts: List[str], 
        batch_size: int = 16
    ) -> np.ndarray:
        all_features = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing texts"):
            batch_texts = texts[i:i + batch_size]
            
            for text in batch_texts:
                embedding = self.get_text_embedding(text)
                all_features.append(embedding)
        
        result = np.array(all_features, dtype=np.float32)
        print(f"Generated {len(texts)} text embeddings with shape: {result.shape}")
        return result

