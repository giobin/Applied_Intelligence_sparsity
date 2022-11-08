import argparse
import math
import os
import time
from datetime import datetime

import torch
from tensor2tensor.utils.bleu_hook import compute_bleu
from torch.nn.utils import prune
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from transformers import BartConfig, BartForConditionalGeneration

import wandb
from utils.feedback import Logger, get_stats
from utils.file_manager import get_dataset
from utils.pruning import select_pruning_params, gamma_decay, get_regularizer_for_extracting_info, sum_masks, crop
from utils.utils import build_vocab, generate_iterator_bucket, get_spec_tokens, \
    init_seeds, count_parameters, bleu_tokenize, pruned, get_configuration, check_max_n_ckpt, get_hyper, \
    get_param_optimizer


def extract(model, device, data_train, optimizer, args, logger,
          parameters_to_prune, step_num_for_epoch, l2_vocab, workdir):
    gamma_cntr = 0
    model.train()
    pad_id, _, _ = get_spec_tokens(l2_vocab)

    for epoch in range(args.epochs):
        for i, inputs in enumerate(data_train, 0):
            gamma_cntr += 1
            inputs = inputs.to(device)

            optimizer.zero_grad()

            loss = model(**inputs)["loss"]
            loss.backward(retain_graph=True)

            info_dict = get_regularizer_for_extracting_info(model, alpha=args.exp_alpha, step=i)
            optimizer.zero_grad()
            break
        break

    return info_dict


def evaluate(model, iterator, device):
    epoch_loss = 0
    model.eval()
    with torch.no_grad():
        for inputs in tqdm(iterator):
            inputs = inputs.to(device)
            loss = model(**inputs)['loss']
            epoch_loss += loss.item()

    model.train()
    return (epoch_loss / len(iterator))


def calculate_bleu(model, iterator, tokenizer, device, args, verbose=None):
    model.eval()
    generated_bleu = []
    trg_bleu = []
    logger = None
    PAD_ID, BOS_ID, EOS_ID = get_spec_tokens(tokenizer)
    beam_size, length_penalty, length = get_hyper(args)
    if verbose is not None: logger = Logger(verbose + "/translated.txt")
    for inputs in tqdm(iterator):
        input_token = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        gen = model.generate(input_token, attention_mask=attention_mask, pad_token_id=PAD_ID, bos_token_id=BOS_ID,
                             eos_token_id=EOS_ID,
                             max_length=length, num_beams=beam_size, length_penalty=length_penalty)
        translated = tokenizer.batch_decode(gen, skip_special_tokens=True)
        target = tokenizer.batch_decode(inputs['labels'], skip_special_tokens=True)

        for elem in translated:
            generated_bleu.append(bleu_tokenize(elem))
        for elem in target:
            trg_bleu.append(bleu_tokenize(elem))

        if logger is not None:
            logger.logging(f'Generated:{translated}', print_=False)
            logger.logging(f'Target:{target}', print_=False)
            logger.logging("\n", print_=False)

    model.train()
    return compute_bleu(trg_bleu, generated_bleu)


def generate_model(tokenizer, args):
    PAD_ID, BOS_ID, EOS_ID = get_spec_tokens(tokenizer)
    vocab_size = len(tokenizer)
    MPE, EL, DL, FFN_DIM, AH, DM, DR = get_configuration(args)

    configuration = BartConfig(vocab_size=vocab_size,
                               activation_function="relu",
                               max_position_embeddings=MPE,
                               encoder_layers=EL,
                               encoder_ffn_dim=FFN_DIM,
                               encoder_attention_heads=AH,
                               decoder_layers=DL,
                               decoder_ffn_dim=FFN_DIM,
                               decoder_attention_heads=AH,
                               pad_token_id=PAD_ID,
                               bos_token_id=BOS_ID,
                               decoder_start_token_id=BOS_ID,
                               eos_token_id=EOS_ID,
                               forced_eos_token_id=EOS_ID,
                               forced_bos_token_id=BOS_ID,
                               d_model=DM,
                               scale_embedding=True,
                               add_bias_logits=False,
                               dropout=DR)
    if args.configuration is not None:
        print("Load Config...")
        path_config = os.path.abspath(os.getcwd()) + "/" + args.configuration
        configuration = configuration.from_json_file(path_config)

    model = BartForConditionalGeneration(configuration)
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)

    if args.ckpt is not None:
        print("Load CKPT...")
        path_model = os.path.abspath(os.getcwd()) + "/" + args.ckpt
        checkpoint_pruned = pruned(path_model)

        if checkpoint_pruned is not None:
            print("Load Pruned CKPT...")
            model_params = select_pruning_params(model)
            for p in model_params:
                prune.identity(p[0], p[1])
            model.load_state_dict(checkpoint_pruned)
            model.tie_weights()
            model.eval()
        else:
            model = model.from_pretrained(path_model, config=configuration)

    return model


def main(args):
    init_seeds()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    workdir = f'{args.work_dir}/{datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}_extracting_info'

    if not os.path.exists(workdir):
        os.makedirs(workdir)
    path_log = workdir + "/log.txt"
    logger = Logger(path_log)

    logger.logging('===================')
    for ar in vars(args):
        logger.logging(f'--{ar}, {getattr(args, ar)}')
    logger.logging('===================')

    # Train Dataset
    path_train = os.path.abspath(os.getcwd()) + "/" + args.train_path
    train_dataset = get_dataset(path_train)
    # Validation Dataset
    path_validation = os.path.abspath(os.getcwd()) + "/" + args.validation_path
    validation_dataset = get_dataset(path_validation)
    # Test Dataset
    path_test = os.path.abspath(os.getcwd()) + "/" + args.test_path
    test_dataset = get_dataset(path_test)
    logger.logging(f'Train:{len(train_dataset)}')
    logger.logging(f'Validation:{len(validation_dataset)}')
    logger.logging(f'Test:{len(test_dataset)}')

    # Generate Vocabs
    path = path_train
    if args.single_vocab:  # One language
        logger.logging('Generating single vocab ...')
        l1_vocab = build_vocab(path + "/train.L1", workdir, "L", path_vocab=args.path_vocab1)
        l2_vocab = l1_vocab
    else:  # Different Languages
        logger.logging('Generating two vocabs ...')
        l1_vocab = build_vocab(path + "/train.L1", workdir, "L1", path_vocab=args.path_vocab1)
        l2_vocab = build_vocab(path + "/train.L2", workdir, "L2", path_vocab=args.path_vocab2)

    logger.logging(f'L1 VOCAB Size:{len(l1_vocab)}')
    logger.logging(f'L2 VOCAB Size:{len(l1_vocab)}')

    step_num_for_epoch = len(train_dataset) / args.batch_size

    # Generate Iterators
    iterator_train = generate_iterator_bucket(train_dataset, l1_vocab, l2_vocab, args)
    iterator_valid = generate_iterator_bucket(validation_dataset, l1_vocab, l2_vocab, args)
    iterator_test = generate_iterator_bucket(test_dataset, l1_vocab, l2_vocab, args)

    # Generate Model
    model = generate_model(l1_vocab, args)
    model.to(device)
    #wandb.watch(model, log="all")
    logger.logging(str(model))

    # Get Pruning Params
    parameters_to_prune = select_pruning_params(model)

    #anyway make it as it was pruned
    for p in parameters_to_prune:
        prune.identity(p[0], p[1])

    num_all_params = count_parameters(model, trainable=False)
    num_trainable_params = count_parameters(model)
    non_zero_params = sum_masks(parameters_to_prune)
    logger.logging(f'there are {len(parameters_to_prune)} parameters_to_prune: {parameters_to_prune}')
    logger.logging(f'num of model parameters: {num_all_params}')
    logger.logging(f'num of trainable model parameters: {num_trainable_params}')
    logger.logging(f'num of non zero parameters: {non_zero_params}')

    # Optimizer and scheduler
    beta1, beta2, eps = get_param_optimizer(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(beta1, beta2), eps=eps)
    if args.opt is not None:
        logger.logging('Loading OPT...')
        opt = torch.load(args.opt)
        optimizer.load_state_dict(opt)
        optimizer.defaults["lr"] = args.learning_rate
        for g in optimizer.param_groups:
            g['lr'] = args.learning_rate

    t_max = int((len(train_dataset) / args.batch_size) * args.epochs)

    # Start Training
    info_dict = extract(model, device, iterator_train, optimizer,
                          args, logger,
                          parameters_to_prune, step_num_for_epoch, l2_vocab, workdir)

    torch.save(info_dict, os.path.join(workdir, "info_dict.pt"))
    print(f"saved info_dict")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-tr', '--train_path', default='data/wmt/train', type=str)
    parser.add_argument('-va', '--validation_path', default='data/wmt/dev', type=str)
    parser.add_argument('-te', '--test_path', default='data/wmt/test', type=str)
    parser.add_argument('--type_dataset', default='wmt', choices=['tsk1', 'wmt'])
    parser.add_argument('-single_vocab', action='store_true')
    parser.add_argument('--path_vocab1', default=None, help="path of vocab1, for ex: workdir/L1.vocab")
    parser.add_argument('--path_vocab2', default=None, help="path of vocab2, for ex: workdir/L2.vocab")
    parser.add_argument('--configuration', default=None,
                        help="path of model configuration, for ex: workdir/config.json")
    parser.add_argument('--ckpt', default=None,
                        help="dir of model ckpt, for ex: workdir/ckpt.bin")
    parser.add_argument('--opt', default=None,
                        help="optimizer state dict path")
    parser.add_argument('--work_dir', type=str, default='workdir', help="dir where to store model ckpt")
    parser.add_argument('-e', '--epochs', type=int, default=10, help="number of epochs to train")
    parser.add_argument('-b', '--batch_size', type=int, default=100, help="batch dimension")
    parser.add_argument('-ga', '--gradient_accumulation', type=int, default=3, help="number of gradient accumulation")
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help="learning rate")
    parser.add_argument('--log_interval', type=int, default=50, help="log interval in number of batches")
    parser.add_argument('--eval_interval', type=int, default=400, help="eval interval in number of batches")
    parser.add_argument('-t', '--threshold', type=float, default=0.10, help="threshold to use to crop weights")
    parser.add_argument('--bleu_lower_bound', type=float, default=0.2720, help="lower acceptable bleu for cropping")
    parser.add_argument('-r', '--regularize', action='store_true', help="regularize")
    parser.add_argument('-rt', '--rtype', type=str, help="the regularization type", default="relevance", choices=['relevance', 'l1', 'l2'])
    parser.add_argument('--save_eval', action='store_true', help="save model after evaluation")
    parser.add_argument('--reg_gamma', type=float, default=10 ** -5, help="regularizer term weight")
    parser.add_argument('--gamma_decay', action='store_true', help="if you want gamma decay or not")
    parser.add_argument('--exp_alpha', type=float, default=1., help="weight to multiply exp argument")
    parser.add_argument('-p', '--patience', type=int, default=5, help="last [patience] checkpoint saved")
    parser.add_argument('--numworkers', type=int, default=8, help="num of data workers")
    args = parser.parse_args()
    wandb.init(project="tsk1_l2", entity='giobin')
    main(args)
