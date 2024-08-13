import torch
from functions import get_loader, get_model
import random
from accelerate import Accelerator
import datetime
import os

# ������������ӣ�ȷ��ʵ��Ŀ��ظ���
def set_random_seed(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ��ʼ����������ģ�͡��Ż�����������
def initialize_components():
    accelerator = Accelerator()
    _, _, loader = get_loader(text_lens=1)
    model, optimizer, scheduler = get_model(num_hidden_layers=4)
    loader, model, optimizer, scheduler = accelerator.prepare(loader, model, optimizer, scheduler)
    return accelerator, loader, model, optimizer, scheduler

# ��ӡ�����������Ϣ
def print_accelerator_info(accelerator):
    print('rank=', os.environ.get('RANK', None))
    print('local_rank=', os.environ.get('LOCAL_RANK', None))
    print('accelerator.distributed_type=', accelerator.distributed_type)
    print('accelerator.is_local_main_process=', accelerator.is_local_main_process)
    print('accelerator.is_main_process=', accelerator.is_main_process)

# ѵ������
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

# ����ģ��
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
