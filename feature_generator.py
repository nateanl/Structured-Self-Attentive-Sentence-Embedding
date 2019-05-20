import torch
from torch.utils.data import Dataset
from torch.autograd import Variable

class Yelp_Dataset(Dataset):

    def __init__(self,mode,cuda):
        """
        Args:
            word_dict: map the word to indices
            pos_dict: map the pos tags to indices
            fn (string): the file to be opened
        """
        self.word_id = torch.load('/scratch/near/anlp/'+mode+'_word_ids.pt')
        self.pos_id = torch.load('/scratch/near/anlp/'+mode+'_pos_ids.pt')
        self.label = torch.load('/scratch/near/anlp/'+mode+'_label.pt')
        self.label = self.label -1
        self.cuda = cuda

    def __len__(self):
        return self.word_id.shape[0]

    def __getitem__(self, idx):
        word_id = Variable(self.word_id[idx])
        pos_id = Variable(self.pos_id[idx])
        label = Variable(self.label[idx])
        return word_id.to(self.cuda), pos_id.to(self.cuda), label.long().to(self.cuda)
