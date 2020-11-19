import random
from data import ImageDetectionsField, TextField, RawField
from data import COCO, DataLoader
import evaluation
from evaluation import PTBTokenizer, Cider
from models.transformer import Transformer, MemoryAugmentedEncoder, MeshedDecoder, ScaledDotProductAttentionMemory
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import NLLLoss
import torch.nn.functional as F
from tqdm import tqdm

import argparse, os, pickle
import numpy as np
import itertools
import multiprocessing
from shutil import copyfile

import os, json


# lines below to make the training reproducible.
seed = 1234
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

torch.cuda.manual_seed_all(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def evaluate_loss(model, dataloader, loss_fn, text_field):
    # Validation loss
    model.eval()
    running_loss = .0
    with tqdm(desc='Epoch %d - validation' % e, unit='it', total=len(dataloader)) as pbar:
        with torch.no_grad():
            for it, (detections, captions) in enumerate(dataloader):
                detections, captions = detections.to(device), captions.to(device)
                out,_ = model(detections, captions)
                captions = captions[:, 1:].contiguous()
                out = out[:, :-1].contiguous()
                loss = loss_fn(out.view(-1, len(text_field.vocab)), captions.view(-1))
                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()

    val_loss = running_loss / len(dataloader)
    return val_loss


def evaluate_metrics(model, dataloader, text_field):
    import itertools
    model.eval()
    gen = {}
    gts = {}
    with tqdm(desc='Epoch %d - evaluation' % e, unit='it', total=len(dataloader)) as pbar:
        for it, (images, caps_gt) in enumerate(iter(dataloader)):
            images = images.to(device)
            with torch.no_grad():
                out, _ = model.beam_search(images, 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)   
            caps_gen = text_field.decode(out, join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i, ]    
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)
    return scores


def train_xe(model, dataloader, dataloader_DA, optim, text_field):
    # Training with cross-entropy (xe) with label smoothing
    data_target_iter = iter(dataloader_DA)
    model.train()
    scheduler.step()
    running_loss = .0
    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader)) as pbar:
        for it, (detections, captions) in enumerate(dataloader):
            detections, captions = detections.to(device), captions.to(device)
            out, domain_output = model(detections, captions)
            optim.zero_grad()
            captions_gt = captions[:, 1:].contiguous()
            out = out[:, :-1].contiguous()
            loss_labelsmoothing = loss_ls_v2(out, captions_gt) 
            domain_label = torch.zeros(detections.size()[0]).long().to(device)
            err_s_domain = loss_domain(domain_output[0:args.grl_batch], domain_label[0:args.grl_batch])

            try:
                data_target = data_target_iter.next()
            except: 
                data_target_iter = iter(dataloader_DA)
                data_target = data_target_iter.next()
            
            t_img, t_cap = data_target
            input_img, t_cap = t_img.to(device), t_cap.to(device) 
            domain_label = torch.ones(input_img.size()[0]).long().to(device)
            _, domain_output = model(input_img, t_cap)
            err_t_domain = loss_domain(domain_output, domain_label)
            err = loss_labelsmoothing + err_s_domain + err_t_domain
            err.backward()


            optim.step()
            this_loss = loss_labelsmoothing.item()
            running_loss += this_loss

            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()
            scheduler.step()

    loss = running_loss / len(dataloader)
    
    return loss


class CELossWithLS(torch.nn.Module):
    def __init__(self, classes=None, smoothing=0.1, gamma=3.0, isCos=True, ignore_index=-1):
        super(CELossWithLS, self).__init__()
        self.complement = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        with torch.no_grad():
            oh_labels = F.one_hot(target.to(torch.int64), num_classes = self.cls).permute(0,1,2).contiguous()
            smoothen_ohlabel = oh_labels * self.complement + self.smoothing / self.cls

        logs = self.log_softmax(logits[target!=self.ignore_index])
        pt = torch.exp(logs)
        return -torch.sum((1-pt).pow(self.gamma)*logs * smoothen_ohlabel[target!=self.ignore_index], dim=1).mean()


if __name__ == '__main__':
    device = torch.device('cuda')
    parser = argparse.ArgumentParser(description='Meshed-Memory Transformer')
    parser.add_argument('--exp_name', type=str, default='m2_transformer')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--m', type=int, default=40)   
    parser.add_argument('--head', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=10000)
    parser.add_argument('--features_path', type=str)
    parser.add_argument('--features_path_DA', type=str)
    parser.add_argument('--annotation_folder', type=str)
    parser.add_argument('--annotation_folder_DA', type=str)
    parser.add_argument('--grl_batch', type=int, default=40)
    args = parser.parse_args()
    print(args)

    print('Training')

    # Pipeline for image regions
    image_field = ImageDetectionsField(detections_path=args.features_path, max_detections=6, load_in_tmp=False)  

   
    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)

    image_field_DA = ImageDetectionsField(detections_path=args.features_path_DA, max_detections=6, load_in_tmp=False) 

    
    # Create the dataset
    dataset = COCO(image_field, text_field, args.features_path, args.annotation_folder, args.annotation_folder)
    dataset_DA = COCO(image_field_DA, text_field, args.features_path_DA, args.annotation_folder_DA, args.annotation_folder_DA)
    train_dataset, val_dataset = dataset.splits 
    train_dataset_DA, val_dataset_DA = dataset_DA.splits  
    
    print("-"*100)
    print('GRL batch:',args.grl_batch)
    print('MICCAI dataset:Train length:',len(train_dataset), 'Valid length:',len(val_dataset))
    print('SGH dataset:All:',len(dataset_DA),'train length:',len(train_dataset_DA), 'Valid length:',len(val_dataset_DA))
    

    if not os.path.isfile('vocab_%s.pkl' % args.exp_name):
        print("Building vocabulary")
        text_field.build_vocab(train_dataset, val_dataset, min_freq=2)  
        pickle.dump(text_field.vocab, open('vocab_%s.pkl' % args.exp_name, 'wb'))
    else:
        text_field.vocab = pickle.load(open('vocab_%s.pkl' % args.exp_name, 'rb'))

    print(len(text_field.vocab))
    print(text_field.vocab.stoi)
   

    # Model and dataloaders
    encoder = MemoryAugmentedEncoder(3, 0, attention_module=ScaledDotProductAttentionMemory, 
                                     attention_module_kwargs={'m': args.m})     
    decoder = MeshedDecoder(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>']) 
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)

    dict_dataset_train = train_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    print(len(dict_dataset_train))   
   
    ref_caps_train = list(train_dataset.text) 
    cider_train = Cider(PTBTokenizer.tokenize(ref_caps_train))
    dict_dataset_val = val_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    print(len(dict_dataset_val))   


    def lambda_lr(s):
        warm_up = args.warmup
        s += 1
        return (model.d_model ** -.5) * min(s ** -.5, s * warm_up ** -1.5)


    # Initial conditions
    optim = Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
    scheduler = LambdaLR(optim, lambda_lr)
    loss_fn = NLLLoss(ignore_index=text_field.vocab.stoi['<pad>'])
    loss_ls_v2 = CELossWithLS(classes=len(text_field.vocab), smoothing=0.1, gamma=0.0, isCos=False, ignore_index=text_field.vocab.stoi['<pad>'])

    loss_domain = torch.nn.NLLLoss()
    use_rl = False
    best_cider = .0
    best_bleu = .0
    best_epoch = 0
    patience = 0
    start_epoch = 0

    

    print("Training starts")

    for e in range(start_epoch, start_epoch + 50):  
        dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
        dataloader_train_DA = DataLoader(dataset_DA, batch_size=args.grl_batch, shuffle=True, num_workers=args.workers, drop_last=True)
        
        dataloader_val = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        
        dict_dataloader_val = DataLoader(dict_dataset_val, batch_size=args.batch_size)
       
        # train model with a word-level cross-entropy loss with label smoothing
        if not use_rl:
            train_loss = train_xe(model, dataloader_train, dataloader_train_DA, optim, text_field)
            
       
        # Validation loss
        val_loss = evaluate_loss(model, dataloader_val, loss_fn, text_field)

        # Validation scores
        scores = evaluate_metrics(model, dict_dataloader_val, text_field)  
        val_cider = scores['CIDEr']


        # Prepare for next epoch
        best = False
        if val_cider >= best_cider:
            best_bleu = scores['BLEU'][0]
            best_cider = val_cider
            best_epoch = e
            best = True

        print("Validation scores", scores, 'Best epoch',best_epoch,'Best bleu:%.4f, cider:%.4f'%(best_bleu,best_cider))

        torch.save({
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'random_rng_state': random.getstate(),
            'epoch': e,
            'val_loss': val_loss,
            'val_cider': val_cider,
            'state_dict': model.state_dict(),
            'optimizer': optim.state_dict(),
            'scheduler': scheduler.state_dict(),
            'patience': patience,
            'best_cider': best_cider,
            'use_rl': use_rl,
        }, 'saved_models/%s_last.pth' % args.exp_name)   

        if best:
            print('saving best epoch...!')
            copyfile('saved_models/%s_last.pth' % args.exp_name, 'saved_models/%s_best.pth' % args.exp_name)

       
    data = torch.load('saved_models/m2_transformer_best.pth')
    model.load_state_dict(data['state_dict'])
    print("Epoch %d" % data['epoch'])  
    print(data['best_cider'])






