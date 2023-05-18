from header import *
from .samplers import DistributedBatchSampler
from .sft_dataset import *

'''
def get_tokenizer(model):
    tokenizer = LlamaTokenizer.from_pretrained(model)
    tokenizer.bos_token_id, tokenizer.eos_token_id = 1, 2
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
'''

def load_sft_dataset(args):
    '''
    tokenizer = get_tokenizer(args['model_path'])
    dataset_name = args['models'][args['model']]['stage1_train_dataset'] # SupervisedDataset, str
    data_path = args["data_path"]
    data = globals()[dataset_name](data_path, tokenizer, args['max_length']) #SupervisedDataset
    '''
    data = SupervisedDataset(args['data_path'], args['image_root_path'])

    sampler = torch.utils.data.RandomSampler(data)
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    batch_size = args['world_size'] * args['dschf'].config['train_micro_batch_size_per_gpu']
    batch_sampler = DistributedBatchSampler(
        sampler, 
        batch_size,
        True,
        rank,
        world_size
    )
    iter_ = DataLoader(
        data, 
        batch_sampler=batch_sampler, 
        num_workers=1,
        collate_fn=data.collate, 
        pin_memory=True
    )
    return data, iter_, sampler
