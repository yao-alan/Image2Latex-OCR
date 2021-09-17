import torch

class LatexDataset(torch.utils.data.Dataset):
    def __init__(self, images, df):
        self.images = images
        self.df = df

    def __getitem__(self, index):
        row = self.df.iloc[index]
        return (
            self.images[(row['index'], row['dataset'])],
            row['label_token_indices']
        )

    def __len__(self):
        return len(self.df)


def gen_dataloader(split, merge_shuffle_df, images, extra_cond=None, 
                   batch_size=5, shuffle=True, merge_train_validate=False):
    # dataset takes value 'train', 'test', 'validate'
    condition = merge_shuffle_df['split'] == split
    if merge_train_validate and split == 'train':
        condition = (merge_shuffle_df['split'] == 'train') \
                    | (merge_shuffle_df['split'] == 'validate')
    if extra_cond is not None:
        condition = condition & extra_cond
    return torch.utils.data.DataLoader(
        dataset=LatexDataset(images=images, df=merge_shuffle_df[condition]),
        batch_size=batch_size, shuffle=shuffle
    )