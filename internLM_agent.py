import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
base_dir = "/root/weights/internlm"
model_dir = snapshot_download(os.path.join(base_dir, 'Shanghai_AI_Laboratory/internlm-chat-7b'))