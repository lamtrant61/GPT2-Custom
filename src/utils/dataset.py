from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, encodings, sep_token_id):
        self.encodings = encodings
        self.sep_token_id = sep_token_id

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        # Lấy input_ids, attention_mask và chuẩn bị labels
        item = {key: val[idx] for key, val in self.encodings.items()}

        # Tìm vị trí của token <|sep|> để tách câu hỏi khỏi câu trả lời
        sep_index = (item['input_ids'] == self.sep_token_id).nonzero(as_tuple=True)[0].item()

        # Gán labels sao cho phần câu hỏi có giá trị -100 (để bỏ qua khi tính loss)
        labels = item['input_ids'].clone()
        labels[:sep_index + 1] = -100  # Bỏ qua phần câu hỏi + token <|sep|>

        item['labels'] = labels  # Gán lại labels

        return item