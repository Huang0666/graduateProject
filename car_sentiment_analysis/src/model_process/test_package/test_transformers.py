import torch
from transformers import AutoTokenizer, AutoModel
import accelerate
from accelerate import Accelerator

def test_transformers_installation():
    print("=== Transformers 和 Accelerate 安装检查 ===")
    
    # 检查版本
    import transformers
    print(f"\n1. 版本信息:")
    print(f"Transformers版本: {transformers.__version__}")
    print(f"Accelerate版本: {accelerate.__version__}")
    
    # 测试tokenizer和模型加载
    print("\n2. 测试模型加载:")
    try:
        print("正在加载BERT分词器...")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        print("分词器加载成功!")
        
        print("\n测试分词器:")
        text = "测试Transformers是否正常工作"
        tokens = tokenizer(text, return_tensors="pt")
        print(f"分词结果: {tokens}")
        
        print("\n正在加载BERT模型...")
        model = AutoModel.from_pretrained("bert-base-chinese")
        print("模型加载成功!")
        
        # 测试模型推理
        print("\n3. 测试模型推理:")
        with torch.no_grad():
            outputs = model(**tokens)
        print(f"模型输出shape: {outputs.last_hidden_state.shape}")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        return
    
    # 测试Accelerator
    print("\n4. 测试Accelerator:")
    accelerator = Accelerator()
    print(f"当前设备: {accelerator.device}")
    print(f"分布式训练: {accelerator.distributed_type}")
    print(f"混合精度: {accelerator.mixed_precision}")
    
    print("\n=== 所有测试完成! ===")

if __name__ == "__main__":
    test_transformers_installation() 