import torch
from functions import DataLoaderModule, ModelModule
import random
from accelerate import Accelerator
import datetime
import os

# 设置随机数种子，确保实验的可重复性
def set_random_seed(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 初始化加速器和模型、优化器、调度器
def initialize_components():
    accelerator = Accelerator()

    # 实例化DataLoaderModule并获取loader
    data_module = DataLoaderModule(text_lens=1)
    _, _, loader = data_module.get_loader()

    # 实例化ModelModule并获取model, optimizer, scheduler
    model_module = ModelModule(num_hidden_layers=4)
    model, optimizer, scheduler = model_module.get_model_components()

    # 使用accelerator准备数据和模型
    loader, model, optimizer, scheduler = accelerator.prepare(loader, model, optimizer, scheduler)
    
    return accelerator, loader, model, optimizer, scheduler

# 打印加速器相关信息
def print_accelerator_info(accelerator):
    print('rank=', os.environ.get('RANK', None))
    print('local_rank=', os.environ.get('LOCAL_RANK', None))
    print('accelerator.distributed_type=', accelerator.distributed_type)
    print('accelerator.is_local_main_process=', accelerator.is_local_main_process)
    print('accelerator.is_main_process=', accelerator.is_main_process)

# 训练过程
def train(loader, model, optimizer, scheduler, accelerator):
    start_time = datetime.datetime.now()
    for i, data in enumerate(loader):
        out = model(**data)
        accelerator.backward(out.loss)
        accelerator.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if i % 1 == 0:
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            labels = data['labels']
            logits = out['logits'].argmax(1)
            acc = (labels == logits).sum().item() / len(labels)
            print(i, len(loader), out.loss.item(), lr, acc, accelerator.device)

    print(f"Training completed in: {datetime.datetime.now() - start_time}")

# 保存模型
def save_model_if_main_process(accelerator, model):
    accelerator.wait_for_everyone()
    if accelerator.is_main_process and accelerator.is_local_main_process:
        print('model.save_pretrained(...)')

def main():
    set_random_seed(0)
    accelerator, loader, model, optimizer, scheduler = initialize_components()
    print_accelerator_info(accelerator)
    train(loader, model, optimizer, scheduler, accelerator)
    save_model_if_main_process(accelerator, model)

if __name__ == '__main__':
    main()
