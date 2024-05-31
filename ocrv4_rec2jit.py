
import torch
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from pytorchocr.base_ocr_v20 import BaseOCRV20
import tools.infer.pytorchocr_utility as utility
from pytorchocr.postprocess import build_post_process
#conda activate paddle

class TextRecognize(BaseOCRV20):
    def __init__(self,  **kwargs):
        weights_path=r"D:\file\ocr_detect\PaddleOCR2Pytorch\torch_weight\ch_ptocr_v4_rec_server_infer.pth"
        yaml_path =r"D:\file\ocr_detect\PaddleOCR2Pytorch\configs\rec\PP-OCRv4\ch_PP-OCRv4_rec_hgnet.yml"
        # rec_image_shape='3,48,320'
        postprocess_params = {
            'name': 'CTCLabelDecode',
            "character_type": 'ch',
            "character_dict_path": 'D:\\file\\ocr_detect\\PaddleOCR2Pytorch\\pytorchocr\\utils\\ppocr_keys_v1.txt',
            "use_space_char": True
        }
        postprocess_op = build_post_process(postprocess_params)
        char_num = len(getattr(postprocess_op, 'character'))
        network_config = utility.AnalysisConfig(weights_path,yaml_path, char_num)
        super(TextRecognize, self).__init__(network_config, **kwargs)
        weights = self.read_pytorch_weights(weights_path)
        self.out_channels = self.get_out_channels(weights)
        self.load_state_dict(weights)
  
        self.net.eval()

    
        # if self.use_gpu:
        #     self.net.cuda()

text_recognize = TextRecognize()
img = torch.randn(1, 3, 48, 160)   #  量化，matmul 



out_onnx_name = "./rec_onnx2qnn_weight/rectoqnn/onnx_t/ocrv4_torch_rec_160.onnx"  #4
torch.onnx.export(
    text_recognize.net,
    img,
    f=out_onnx_name,
    # input_names=["img"],
    # output_names=["logits", "boxes"],
    opset_version=12,
    do_constant_folding=True,
    verbose=True)
print("export onnx ok!")


# traced_model = torch.jit.trace(model, (img, input_ids, attention_mask, position_ids, token_type_ids, text_token_mask),strict=False)
# traced_model = torch.jit.trace(text_detector.net,(img),strict=False) 
# 对于qnn ，不能用字典

# traced_model = torch.jit.trace(text_recognize.net,(img),strict=True)  #一种是torch.jit.trace，另一种是torch.jit.script  script能够保存整个的网络结构，而trace只能保存你输入张量经过的网络结构
# torch.jit.save(traced_model, r'D:\file\ocr_detect\PaddleOCR2Pytorch\rec_onnx2qnn_weight\jit_ocrv4_rec.pt')

# print("ok!")