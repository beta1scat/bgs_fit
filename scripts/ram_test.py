# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base') 
#This is the tokenizer used for the above model you have mentioned facebook/xmod-base


from ram.models import ram, ram_plus
from ram import inference_ram

ram_checkpoint = "/root/ws_host/src/data/models/ram_swin_large_14m.pth"
device_cpu = "cpu"
ram_model = ram(pretrained=ram_checkpoint, image_size=384, vit='swin_l')
ram_model.eval()
ram_model = ram_model.to(device_cpu)
print("end")
