import torch
import random
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
from transformers.optimization import get_scheduler
from datasets import Dataset

class DataLoaderModule:
    def __init__(self, vocab_file='tokenizer/vocab.txt', model_max_length=512, text_lens=100, batch_size=32, num_samples=2000):
        self.tokenizer = self._create_tokenizer(vocab_file, model_max_length)
        self.dataset = self._generate_dataset(text_lens, num_samples)
        self.loader = self._create_dataloader(self.dataset, batch_size)

    def _create_tokenizer(self, vocab_file, model_max_length):
        return BertTokenizer(vocab_file=vocab_file, model_max_length=model_max_length)

    def _generate_dataset(self, text_lens, num_samples):
        def data_generator():
            for _ in range(num_samples):
                label = random.randint(0, 9)
                text = ' '.join(str(label) * text_lens)
                yield {'text': text, 'label': label}

        return Dataset.from_generator(data_generator)

    def _collate_batch(self, data):
        texts = [item['text'] for item in data]
        labels = [item['label'] for item in data]

        tokenized_data = self.tokenizer(texts,
                                        padding=True,
                                        truncation=True,
                                        max_length=512,
                                        return_tensors='pt')

        tokenized_data['labels'] = torch.LongTensor(labels)
        return tokenized_data

    def _create_dataloader(self, dataset, batch_size):
        return torch.utils.data.DataLoader(dataset=dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           drop_last=True,
                                           collate_fn=self._collate_batch)

    def get_loader(self):
        return self.tokenizer, self.dataset, self.loader


class ModelModule:
    def __init__(self, num_hidden_layers=32, num_labels=10, lr=1e-4, num_training_steps=50):
        self.model, self.optimizer, self.scheduler = self._create_model(num_hidden_layers, num_labels, lr, num_training_steps)

    def _create_model(self, num_hidden_layers, num_labels, lr, num_training_steps):
        config = BertConfig(num_labels=num_labels, num_hidden_layers=num_hidden_layers)
        model = BertForSequenceClassification(config)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = get_scheduler(name='cosine',
                                  num_warmup_steps=0,
                                  num_training_steps=num_training_steps,
                                  optimizer=optimizer)

        return model, optimizer, scheduler

    def get_model_components(self):
        return self.model, self.optimizer, self.scheduler


# 使用示例
if __name__ == '__main__':
    # 初始化数据加载模块
    data_module = DataLoaderModule()
    tokenizer, dataset, loader = data_module.get_loader()

    # 初始化模型模块
    model_module = ModelModule()
    model, optimizer, scheduler = model_module.get_model_components()

 
