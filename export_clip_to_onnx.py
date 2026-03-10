import torch
import clip
import onnxruntime as ort
import numpy as np

from torchvision import transforms
from utils.evaluation import CallBackVerification


# --------------------------------------------------
# ONNX Wrapper
# --------------------------------------------------
class ONNXModelWrapper(torch.nn.Module):
    def __init__(self, onnx_path: str, device: str = "cpu"):
        super().__init__()
        self.device = device

        providers = (
            ["CUDAExecutionProvider"]
            if device.startswith("cuda")
            else ["CPUExecutionProvider"]
        )

        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def forward(self, x: torch.Tensor):

        x_np = x.detach().cpu().numpy().astype(np.float32)
        output = self.session.run(
            [self.output_name], {self.input_name: x_np}
        )[0]

        return torch.from_numpy(output).to(self.device)


# --------------------------------------------------
# Transform
# --------------------------------------------------
def build_transform(image_size):

    mean_clip = (0.48145466, 0.4578275, 0.40821073)
    std_clip = (0.26862954, 0.26130258, 0.27577711)

    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(
                image_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_clip, std=std_clip),
        ]
    )


# --------------------------------------------------
# Load CLIP Backbone
# --------------------------------------------------
def load_clip_visual(backbone_name: str, device: str):

    backbone, _ = clip.load(backbone_name, device=device, jit=False)

    for param in backbone.parameters():
        if param.dtype == torch.float16:
            param.data = param.data.float()

    model = backbone.visual
    model.eval()

    return model


# --------------------------------------------------
# Export ONNX
# --------------------------------------------------
def export_onnx(model, device, output_path, image_size):

    dummy_input = torch.randn(
        1, 3, image_size, image_size, device=device, dtype=torch.float32
    )

    torch.onnx.export(
        model=model,
        args=dummy_input,
        f=output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():

    # Configuration
    rank = 0
    device = "cuda"
    image_size = 224
    batch_size = 256
    backbone_name = "ViT-B/16"

    val_targets = ["lfw"]  # List of validation datasets to evaluate
    eval_path = "TODO" # Path to the evaluation data directory
    onnx_path = "clip_visual.onnx"

    # Transform
    transform = build_transform(image_size)

    # Evaluation callback
    callback_verification = CallBackVerification(
        5,
        rank,
        val_targets,
        eval_path,
        image_size,
        transform,
        batch_size,
        "",
    )

    # Load CLIP vision encoder
    model = load_clip_visual(backbone_name, device)

    # Evaluate CLIP model
    print("Evaluating CLIP model...")
    callback_verification(4, model)

    # Export ONNX
    print("Exporting ONNX model...")
    export_onnx(model, device, onnx_path, image_size)

    # Evaluate ONNX model
    print("Evaluating ONNX model...")
    onnx_model = ONNXModelWrapper(onnx_path, device=device)
    callback_verification(4, onnx_model)


if __name__ == "__main__":
    main()