# Load model directly
from transformers import AutoModel, AutoModelForCausalLM
model = AutoModel.from_pretrained("openbmb/MiniCPM-Llama3-V-2_5", trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained("openbmb/MiniCPM-Llama3-V-2_5")

import torch.quantization

model_fp32 = model
model_fp32.eval()
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32, {torch.nn.Linear}, dtype=torch.qint8
)