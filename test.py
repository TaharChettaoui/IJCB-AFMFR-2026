import torch
import clip
import onnxruntime as ort
import numpy as np

from utils.evaluation import CallBackVerification
from utils.utils import build_transform

# --------------------------------------------------
# ONNX Wrapper
# --------------------------------------------------
class ONNXModelWrapper(torch.nn.Module):
    def __init__(self, onnx_path: str, device: str = "cpu"):
        super().__init__()
        self.device = device
        print(f"Running on device: {self.device}")

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
        output = self.session.run([self.output_name], {self.input_name: x_np})[0]

        return torch.from_numpy(output).to(self.device)

# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    # Configuration
    rank = 0
    device = "cuda"
    image_size = 224
    batch_size = 256
    val_targets = ["lfw"]  # List of validation datasets to evaluate
    eval_path = "/home/chettaou/workspace/data/validation" # Path to the evaluation data directory
    onnx_path = "clip_visual.onnx"

    # Transform
    MEAN_CLIP = (0.48145466, 0.4578275, 0.40821073)
    STD_CLIP = (0.26862954, 0.26130258, 0.27577711)
    transform = build_transform(image_size, mean=MEAN_CLIP, std=STD_CLIP)

    # Evaluation callback
    callback_verification = CallBackVerification(
        5,
        rank,
        val_targets,
        eval_path,
        image_size,
        transform,
        batch_size,
    )

    # Evaluate ONNX model
    print("Evaluating ONNX model...")
    onnx_model = ONNXModelWrapper(onnx_path, device=device)
    callback_verification(4, onnx_model)


if __name__ == "__main__":
    main()