from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        # Set labels as the input_ids for language modeling
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = item['input_ids']  # Set labels for loss calculation
        return item