import torch
print(torch.cuda.is_available())       
print(torch.cuda.get_device_name(0))    

# incase GPU doesnt work 
'''
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --> change the cu based on your GPU
i use cu121 for my 3060 

'''
