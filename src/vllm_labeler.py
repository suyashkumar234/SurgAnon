"""
SurgVLP-based zero-shot OoB classifier for surgical video frames.
CLIP-style surgical vision-language model (PeskaVLP variant).
Weights: https://github.com/camma-public/surgvlp
"""

from pathlib import Path

_SURGVLP_TEXTS = {
    "in-body": [
        "da vinci robotic instruments manipulating tissue inside the body during surgery",
        "robotic laparoscopic instruments dissecting tissue inside the patient's abdomen",
    ],
    "out-of-body": [
        "a doctor's or surgeon's face is visible in the camera view",
        "patient skin or body exterior is visible outside the surgical site",
    ],
}

_PESKA_CONFIG = {
    "type": "PeskaVLP",
    "backbone_img": {
        "type": "img_backbones/ImageEncoder",
        "num_classes": 768,
        "pretrained": "random",
        "backbone_name": "resnet_50",
        "img_norm": True,
    },
    "backbone_text": {
        "type": "text_backbones/BertEncoder",
        "text_bert_type": "emilyalsentzer/Bio_ClinicalBERT",
        "text_last_n_layers": 1,
        "text_aggregate_method": "mean",
        "text_norm": True,
        "text_embedding_dim": 768,
        "text_freeze_bert": False,
        "text_agg_tokens": False,
    },
}


class SurgVLPLabeler:
    """Zero-shot OoB classifier using SurgVLP (PeskaVLP variant)."""

    def __init__(self, ckpt_path: str | None = None, device: str | None = None):
        import torch
        import surgvlp

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._surgvlp = surgvlp

        ckpt_path = ckpt_path or self._find_or_download_weights()

        print(f"  Loading SurgVLP (PeskaVLP) from {ckpt_path} on {device} …")
        self.model, self.preprocess = surgvlp.load(
            _PESKA_CONFIG, device=device, pretrain=ckpt_path
        )
        self.model.eval()

        print("  Computing text embeddings (prompt ensemble) …")
        with torch.no_grad():
            self._text_features = {}
            for label, texts in _SURGVLP_TEXTS.items():
                feats = []
                for text in texts:
                    tokens = surgvlp.tokenize(text, device=device)
                    feat = self.model(inputs_text=tokens, mode="text")["text_emb"]
                    feats.append(feat / feat.norm(dim=-1, keepdim=True))
                avg_feat = torch.stack(feats).mean(0)
                avg_feat = avg_feat / avg_feat.norm(dim=-1, keepdim=True)
                self._text_features[label] = avg_feat.squeeze(0)
        print("  SurgVLP ready.")

    def _find_or_download_weights(self) -> str:
        cache_dir = Path.home() / ".cache" / "surgvlp"
        cache_dir.mkdir(parents=True, exist_ok=True)

        extracted = cache_dir / "PeskaVLP.pth"
        if extracted.exists():
            return str(extracted)

        zipped = cache_dir / "peska_vlp.pth"
        if zipped.exists():
            import zipfile
            print("  Extracting PeskaVLP weights …")
            with zipfile.ZipFile(str(zipped), "r") as z:
                z.extractall(str(cache_dir))
            if extracted.exists():
                return str(extracted)

        print("  Downloading PeskaVLP weights …")
        from surgvlp.surgvlp import _download, _MODELS
        return _download(_MODELS, "PeskaVLP", str(cache_dir))

    def label_frame(self, image_path: str) -> dict:
        import torch
        import math
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            img_feat = self.model(inputs_img=image_tensor, mode="video")["img_emb"]
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        scores = {
            label: float((img_feat @ txt_feat.T).squeeze())
            for label, txt_feat in self._text_features.items()
        }
        exp_scores = {k: math.exp(v * 100) for k, v in scores.items()}
        total = sum(exp_scores.values())
        probs = {k: v / total for k, v in exp_scores.items()}

        best_label = max(probs, key=probs.get)
        return {
            "label":      best_label,
            "confidence": round(probs[best_label], 4),
            "reason":     f"in-body={probs['in-body']:.3f} oob={probs['out-of-body']:.3f}",
        }

    def label_frames(self, image_paths: list[str], batch_size: int = 32) -> list[dict]:
        results = []
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i: i + batch_size]
            for path in batch:
                results.append(self.label_frame(path))
            done = min(i + batch_size, len(image_paths))
            if done % 100 == 0 or done == len(image_paths):
                print(f"    [{done}/{len(image_paths)}] labeled")
        return results
