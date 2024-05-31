import os

from onnxruntime.quantization import quantize_dynamic, QuantType,quantize_static
# from onnxconverter_common import auto_mixed_precision
import onnx

def quantize(path):
    """
    Quantize the weights of the model from float32 to in8 to allow very efficient inference on
    modern CPU

    Uses unsigned ints for activation values, signed ints for weights, per
    https://onnxruntime.ai/docs/performance/quantization.html#data-type-selection
    it is faster on most CPU architectures

    Args:
        path: Path to location the exported ONNX model is stored

    Returns:
        The Path generated for the quantized
    """
    print("Quantizing...")
    outpath = path[:-5] + "_int8.onnx"
    quantize_dynamic(
        model_input=path,
        model_output=outpath,
        per_channel=True,
        reduce_range=True, 
        weight_type=QuantType.QInt8,  # per docs, signed is faster on most CPUs
        # weight_type=QuantType.QUInt16,  # 运行报错
        # weight_type=QuantType.QInt16,  # 运行报错
       # optimize_model=True,
    #    op_types_to_quantize=['MatMul', 'Relu', 'Add', 'Mul'],


    )  # op_types_to_quantize=['MatMul', 'Relu', 'Add', 'Mul'],
    # os.remove(path[:-5] + "-opt.onnx")
    print("Done")


def quantize_0(path):
    # model_fp16 = auto_convert_mixed_precision(model, test_data, rtol=0.01, atol=0.001, keep_io_types=True)
    # onnx.save(model_fp16, "path/to/model_fp16.onnx")
    pass


if __name__=="__main__":

    # ori_onnx = "./output/decoder.onnx"
    # ori_onnx = "./output/encoder.onnx"
    # ori_onnx = "./output/decoder_pkv.onnx"
    # ori_onnx = "./output/lm_head.opt.onnx"
    ori_onnx=r"D:\file\ocr_detect\PaddleOCR\paddle_weight\ch-pp-ocrv4-server\ch_PP-OCRv4_rec_server_infer\model16.onnx"

    # ori_onnx = "./output/newtest/encoder.onnx"
    # ori_onnx = "./output/newtest/decoder_pkv.onnx"
    # ori_onnx = "./output/newtest/decoder.onnx"
    quantize(ori_onnx)
    print("ok")
