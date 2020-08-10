from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
import argparse
import json
import logging
import os
import random
import time
import torch
import torch.nn as nn

from models import IQ
from models import gen_ques_rnn
from utils import Vocabulary
from utils import get_glove_embedding
from utils import get_loader
from utils import load_vocab
from utils import process_lengths
from utils import gaussian_KL_loss

from center_loss import CenterLoss

def create_model(args, vocab, embedding=None):
    if args.use_glove:
        embedding = get_glove_embedding(args.embedding_name,
                                        args.hidden_size,
                                        vocab)
    else:
        embedding = None

    logging.info('Creating IQ model...')

    vqg = IQ(len(vocab), args.max_length, args.hidden_size,
             args.num_categories,
             vocab(vocab.SYM_SOQ), vocab(vocab.SYM_EOS),
             num_layers=args.num_layers,
             rnn_cell=args.rnn_cell,
             dropout_p=args.dropout_p,
             input_dropout_p=args.input_dropout_p,
             encoder_max_len=args.encoder_max_len,
             embedding=embedding,
             num_att_layers=args.num_att_layers,
             z_size=args.z_size,
             z_img=args.z_img,
             z_category=args.z_category, 
             no_image_recon=args.no_image_recon,
             no_category_space=args.no_category_space, bayes=args.bayes)

    return vqg


def evaluate(vqg, data_loader, criterion, l2_criterion, args):
    vqg.eval()
    
    if(args.bayes):
        alpha = vqg.alpha
    
    total_gen_loss = 0.0
    total_kl = 0.0
    total_recon_image_loss = 0.0
    total_recon_category_loss = 0.0
    total_z_t_kl = 0.0
    total_t_kl_loss = 0.0
    regularisation_loss = 0.0
    c_loss = 0.0
    category_cycle_loss = 0.0

    total_steps = len(data_loader)
    if args.eval_steps is not None:
        total_steps = min(len(data_loader), args.eval_steps)
    start_time = time.time()
    for iterations, (images, questions, answers,
            categories, qindices) in enumerate(data_loader):

        ''' remove answers from the dataloader later '''

        # Set mini-batch dataset
        if torch.cuda.is_available():
            images = images.cuda()
            questions = questions.cuda()
            answers = answers.cuda()
            categories = categories.cuda()
            qindices = qindices.cuda()
            if(args.bayes):
                alpha = vqg.alpha.cuda()
        
        # Forward, Backward and Optimize
        image_features = vqg.encode_images(images)
        category_features = vqg.encode_categories(categories)
        t_mus, t_logvars, ts = vqg.encode_into_t(image_features, category_features)
        (outputs, _, other), pred_ques = vqg.decode_questions(
                image_features, ts, questions=questions,
                teacher_forcing_ratio=1.0)

        # Reorder the questions based on length.
        questions = torch.index_select(questions, 0, qindices)
        total_loss = 0.0

        # Ignoring the start token.
        questions = questions[:, 1:]
        qlengths = process_lengths(questions)

        # Convert the output from MAX_LEN list of (BATCH x VOCAB) ->
        # (BATCH x MAX_LEN x VOCAB).
        outputs = [o.unsqueeze(1) for o in outputs]
        outputs = torch.cat(outputs, dim=1)
        outputs = torch.index_select(outputs, 0, qindices)

        if(args.step_two):
            category_cycle = vqg.encode_questions(pred_ques, qlengths)
            category_cycle_loss = criterion(category_cycle, categories)
        
            category_cycle_loss = category_cycle_loss.item()
            total_loss += args.lambda_c_cycle * category_cycle_loss

        # Calculate the loss.
        targets = pack_padded_sequence(questions, qlengths,
                                       batch_first=True)[0]
        outputs = pack_padded_sequence(outputs, qlengths,
                                       batch_first=True)[0]
        
        gen_loss = criterion(outputs, targets)
        total_gen_loss += gen_loss.data.item()

        # Get KL loss if it exists.
        if(args.bayes):
            regularisation_loss = l2_criterion(alpha.pow(-1), torch.ones_like(alpha))
            kl_loss = -0.5 * torch.sum(1 + t_logvars + alpha.pow(2).log() - alpha.pow(2) * ( t_mus.pow(2) + t_logvars.exp()))
            total_kl += kl_loss.item() + regularisation_loss.item()
        else:
            kl_loss = gaussian_KL_loss(t_mus, t_logvars)
            total_kl += args.lambda_t * kl_loss
            kl_loss = kl_loss.item()

        # Reconstruction.
        if not args.no_image_recon or not args.no_category_space:
            image_targets = image_features.detach()
            category_targets = category_features.detach()
            
            recon_image_features, recon_category_features = vqg.reconstruct_inputs(image_targets, category_targets)
            
            if not args.no_image_recon:
                recon_i_loss = l2_criterion(recon_image_features, image_targets)
                total_recon_image_loss += recon_i_loss.item()
            
            if not args.no_category_space:
                recon_c_loss = l2_criterion(recon_category_features, category_targets)
                total_recon_category_loss += recon_c_loss.item()

        # Quit after eval_steps.
        if args.eval_steps is not None and iterations >= args.eval_steps:
            break

        # Print logs
        if iterations % args.log_step == 0:
             delta_time = time.time() - start_time
             start_time = time.time()
             logging.info('Time: %.4f, Step [%d/%d], gen loss: %.4f, '
                          'KL: %.4f, I-recon: %.4f, C-recon: %.4f, C-cycle: %.4f, Regularisation: %.4f'
                         % (delta_time, iterations, total_steps,
                            total_gen_loss/(iterations+1),
                            total_kl/(iterations+1),
                            total_recon_image_loss/(iterations+1),
                            total_recon_category_loss/(iterations+1),
                            category_cycle_loss/(iterations+1),
                            regularisation_loss/(iterations+1)))

    total_info_loss = total_recon_image_loss + total_recon_category_loss
    return total_gen_loss / (iterations+1), total_info_loss / (iterations + 1)


def run_eval(vqg, data_loader, criterion, l2_criterion, args, epoch,
             scheduler, info_scheduler):
    logging.info('=' * 80)
    start_time = time.time()
    val_gen_loss, val_info_loss = evaluate(
            vqg, data_loader, criterion, l2_criterion, args)
    delta_time = time.time() - start_time
    scheduler.step(val_gen_loss)
    scheduler.step(val_info_loss)
    logging.info('Time: %.4f, Epoch [%d/%d], Val-gen-loss: %.4f, '
                 'Val-info-loss: %.4f' % (
        delta_time, epoch, args.num_epochs, val_gen_loss, val_info_loss))
    logging.info('=' * 80)


def sample_for_each_category(vqg, image, args):
    """Sample a question per category.

    Args:
        vqg: Question generation model.
        image: The image for which to generate questions for.
        args: Instance of ArgumentParser.

    Returns:
        A list of questions per category.
    """
    if args.no_category_space:
        return None
    categories = torch.LongTensor(range(args.num_categories))
    if torch.cuda.is_available():
        categories = categories.cuda()
    images = image.unsqueeze(0).expand((
        args.num_categories, image.size(0), image.size(1), image.size(2)))
    outputs = vqg.predict_from_category(images, categories)
    return outputs


def compare_outputs(images, questions, answers, categories,
                    vqg, vocab, logging, cat2name,
                    args, num_show=5):
    """Sanity check generated output as we train.

    Args:
        images: Tensor containing images.
        questions: Tensor containing questions as indices.
        answers: Tensor containing answers as indices.
        categories: Tensor containing categories as indices.
        alengths: list of answer lengths.
        vqg: A question generation instance.
        vocab: An instance of Vocabulary.
        logging: logging to use to report results.
        cat2name: Mapping from category index to answer type name.
    """
    vqg.eval()

    # Forward pass through the model.
    outputs = vqg.predict_from_category(images, categories)

    for _ in range(num_show):
        logging.info("         ")
        i = random.randint(0, images.size(0) - 1)  # Inclusive.

        # Sample some types.
        if not args.no_category_space:
            category_outputs = vqg.predict_from_category(images, categories)
            category_question = vocab.tokens_to_words(category_outputs[i])
            logging.info('Typed question: %s' % category_question)
            category_checks = sample_for_each_category(vqg, images[i], args)
            category_checks = [cat2name[idx] + ': ' + vocab.tokens_to_words(category_checks[j])
                           for idx, j in enumerate(range(category_checks.size(0)))]
            logging.info('category checks: ' + ', '.join(category_checks))

        # Log the outputs.
        output = vocab.tokens_to_words(outputs[i])
        question = vocab.tokens_to_words(questions[i])
        category = categories[i]
        logging.info('Sampled question : %s\n'
                     'Target  question (%s): %s -> %s'
                     % (output, cat2name[categories[i].item()],
                        question, category))
        logging.info("         ")

def train(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Save the arguments.
    with open(os.path.join(args.model_path, 'args.json'), 'w') as args_file:
        json.dump(args.__dict__, args_file)

    # Config logging.
    log_format = '%(levelname)-8s %(message)s'
    log_file_name = 'train_'+args.train_log_file_suffix+'.log'
    logfile = os.path.join(args.model_path, log_file_name)
    logging.basicConfig(filename=logfile, level=logging.INFO, format=log_format)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(json.dumps(args.__dict__))

    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(args.crop_size,
                                     scale=(1.00, 1.2),
                                     ratio=(0.75, 1.3333333333333333)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    # Load vocabulary wrapper.
    vocab = load_vocab(args.vocab_path)

    # Load the category types.
    cat2name = json.load(open(args.cat2name))

    # Build data loader
    logging.info("Building data loader...")
    train_sampler = None
    val_sampler = None
    if os.path.exists(args.train_dataset_weights):
        train_weights = json.load(open(args.train_dataset_weights))
        train_weights = torch.DoubleTensor(train_weights)
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(
                train_weights, len(train_weights))

    if os.path.exists(args.val_dataset_weights):
        val_weights = json.load(open(args.val_dataset_weights))
        val_weights = torch.DoubleTensor(val_weights)
        val_sampler = torch.utils.data.sampler.WeightedRandomSampler(
                val_weights, len(val_weights))

    data_loader = get_loader(args.dataset, transform,
                                 args.batch_size, shuffle=False,
                                 num_workers=args.num_workers,
                                 max_examples=args.max_examples,
                                 sampler=train_sampler)
    val_data_loader = get_loader(args.val_dataset, transform,
                                     args.batch_size, shuffle=False,
                                     num_workers=args.num_workers,
                                     max_examples=args.max_examples,
                                     sampler=val_sampler)

    print('Done loading data ............................')

    logging.info("Done")

    vqg = create_model(args, vocab)
    if args.load_model is not None:
        vqg.load_state_dict(torch.load(args.load_model))
    
    logging.info("Done")

    # Loss criterion.
    pad = vocab(vocab.SYM_PAD)  # Set loss weight for 'pad' symbol to 0
    criterion = nn.CrossEntropyLoss()
    criterion2 = nn.MultiMarginLoss().cuda()
    l2_criterion = nn.MSELoss()

    alpha = None

    if(args.bayes):
        alpha = vqg.alpha

    # Setup GPUs.
    if torch.cuda.is_available():
        logging.info("Using available GPU...")
        vqg.cuda()
        criterion.cuda()
        l2_criterion.cuda()
        if(alpha is not None):
            alpha.cuda()

    gen_params = vqg.generator_parameters()
    info_params = vqg.info_parameters()

    learning_rate = args.learning_rate
    info_learning_rate = args.info_learning_rate
    
    gen_optimizer = torch.optim.Adam(gen_params, lr=learning_rate)
    info_optimizer = torch.optim.Adam(info_params, lr=info_learning_rate)

    if(args.step_two):
        cycle_params = vqg.cycle_params()
        cycle_optimizer = torch.optim.Adam(cycle_params, lr=learning_rate)

    if(args.center_loss):
        center_loss = CenterLoss(num_classes=args.num_categories, feat_dim=args.z_size, use_gpu=True)
        optimizer_centloss = torch.optim.SGD(center_loss.parameters(), lr=0.5)
    
    scheduler = ReduceLROnPlateau(optimizer=gen_optimizer, mode='min',
                                  factor=0.5, patience=args.patience,
                                  verbose=True, min_lr=1e-7)
    cycle_scheduler = ReduceLROnPlateau(optimizer=gen_optimizer, mode='min',
                                  factor=0.99, patience=args.patience,
                                  verbose=True, min_lr=1e-7)
    info_scheduler = ReduceLROnPlateau(optimizer=info_optimizer, mode='min',
                                       factor=0.5, patience=args.patience,
                                       verbose=True, min_lr=1e-7)

    # Train the model.
    total_steps = len(data_loader)
    start_time = time.time()
    n_steps = 0

    # Optional losses. Initialized here for logging.
    recon_category_loss = 0.0
    recon_image_loss = 0.0
    kl_loss = 0.0
    category_cycle_loss = 0.0
    regularisation_loss = 0.0
    c_loss = 0.0
    cycle_loss = 0.0
    
    if(args.step_two):
        category_cycle_loss = 0.0
    
    if(args.bayes):
        regularisation_loss = 0.0

    if(args.center_loss):
        loss_center = 0.0
        c_loss = 0.0

    for epoch in range(args.num_epochs):
        for i, (images, questions, answers,
                categories, qindices) in enumerate(data_loader):
            n_steps += 1

            ''' remove answers from dataloader later '''

            # Set mini-batch dataset.
            if torch.cuda.is_available():
                images = images.cuda()
                questions = questions.cuda()
                answers = answers.cuda()
                categories = categories.cuda()
                qindices = qindices.cuda()
                if(args.bayes):
                    alpha = alpha.cuda()
            
            # Eval now.
            if (args.eval_every_n_steps is not None and
                    n_steps >= args.eval_every_n_steps and
                    n_steps % args.eval_every_n_steps == 0):
                run_eval(vqg, val_data_loader, criterion, l2_criterion,
                         args, epoch, scheduler, info_scheduler)
                compare_outputs(images, questions, answers, categories,
                                vqg, vocab, logging, cat2name, args)

            # Forward.
            vqg.train()
            gen_optimizer.zero_grad()
            info_optimizer.zero_grad()
            if(args.step_two):
                cycle_optimizer.zero_grad()
            if(args.center_loss):
                optimizer_centloss.zero_grad()
            
            image_features = vqg.encode_images(images)
            category_features = vqg.encode_categories(categories)
            
            # Question generation.
            t_mus, t_logvars, ts = vqg.encode_into_t(image_features, category_features)
            
            if(args.center_loss):
                loss_center = 0.0
                c_loss = center_loss(ts, categories)
                loss_center += args.lambda_centerloss *  c_loss
                c_loss = c_loss.item()
                loss_center.backward(retain_graph=True)
            
                for param in center_loss.parameters():
                    param.grad.data *= (1. / args.lambda_centerloss)

                optimizer_centloss.step()

            qlengths_prev = process_lengths(questions)

            (outputs, _, _), pred_ques = vqg.decode_questions(
                    image_features, ts, questions=questions,
                    teacher_forcing_ratio=1.0)

            # Reorder the questions based on length.
            questions = torch.index_select(questions, 0, qindices)

            # Ignoring the start token.
            questions = questions[:, 1:]
            qlengths = process_lengths(questions)

            # Convert the output from MAX_LEN list of (BATCH x VOCAB) ->
            # (BATCH x MAX_LEN x VOCAB).
            
            outputs = [o.unsqueeze(1) for o in outputs]
            outputs = torch.cat(outputs, dim=1)
            
            outputs = torch.index_select(outputs, 0, qindices)

            if(args.step_two):
                category_cycle_loss = 0.0
                category_cycle = vqg.encode_questions(pred_ques, qlengths)
                cycle_loss = criterion(category_cycle, categories)
                category_cycle_loss += args.lambda_c_cycle * cycle_loss 
                cycle_loss = cycle_loss.item()
                category_cycle_loss.backward(retain_graph=True)
                cycle_optimizer.step()

            # Calculate the generation loss.
            targets = pack_padded_sequence(questions, qlengths,
                                           batch_first=True)[0]
            outputs = pack_padded_sequence(outputs, qlengths,
                                           batch_first=True)[0]

            gen_loss = criterion(outputs, targets)
            total_loss = 0.0
            total_loss += args.lambda_gen * gen_loss
            gen_loss = gen_loss.item()

            # Variational loss.
            if(args.bayes):
                kl_loss = -0.5 * torch.sum(1 + t_logvars + alpha.pow(2).log() - alpha.pow(2) * ( t_mus.pow(2) + t_logvars.exp()))
                regularisation_loss = l2_criterion(alpha.pow(-1), torch.ones_like(alpha))
                total_loss += args.lambda_t * kl_loss + args.lambda_reg * regularisation_loss
                kl_loss = kl_loss.item()
                regularisation_loss = regularisation_loss.item()
            else:
                kl_loss = gaussian_KL_loss(t_mus, t_logvars)
                total_loss += args.lambda_t * kl_loss
                kl_loss = kl_loss.item()

            # Generator Backprop.
            total_loss.backward(retain_graph=True)
            gen_optimizer.step()

            # Reconstruction loss.
            recon_image_loss = 0.0
            recon_category_loss = 0.0

            if not args.no_category_space or not args.no_image_recon:
                total_info_loss = 0.0
                category_targets = category_features.detach()
                image_targets = image_features.detach()
                recon_image_features, recon_category_features = vqg.reconstruct_inputs(
                        image_targets, category_targets)

                # Category reconstruction loss.
                if not args.no_category_space:
                    recon_c_loss = l2_criterion(recon_category_features, category_targets)           # changed to criterion2
                    total_info_loss += args.lambda_c * recon_c_loss
                    recon_category_loss = recon_c_loss.item()

                # Image reconstruction loss.
                if not args.no_image_recon:
                    recon_i_loss = l2_criterion(recon_image_features, image_targets)
                    total_info_loss += args.lambda_i * recon_i_loss
                    recon_image_loss = recon_i_loss.item()

                # Info backprop.
                total_info_loss.backward()
                info_optimizer.step()

            # Print log info
            if i % args.log_step == 0:
                delta_time = time.time() - start_time
                start_time = time.time()
                logging.info('Time: %.4f, Epoch [%d/%d], Step [%d/%d], '
                             'LR: %f, Center-Loss: %.4f, KL: %.4f, '
                             'I-recon: %.4f, C-recon: %.4f, C-cycle: %.4f, Regularisation: %.4f'
                             % (delta_time, epoch, args.num_epochs, i,
                                total_steps, gen_optimizer.param_groups[0]['lr'],
                                c_loss, kl_loss,
                                recon_image_loss, recon_category_loss, cycle_loss, regularisation_loss))

            # Save the models
            if args.save_step is not None and (i+1) % args.save_step == 0:
                torch.save(vqg.state_dict(),
                           os.path.join(args.model_path,
                                        'vqg-tf-%d-%d.pkl'
                                        % (epoch + 1, i + 1)))

        torch.save(vqg.state_dict(),
                   os.path.join(args.model_path,
                                'vqg-tf-%d.pkl' % (epoch+1)))

        torch.save(center_loss.state_dict(),
                           os.path.join(args.model_path,
                                        'closs-tf-%d-%d.pkl'
                                        % (epoch + 1, i + 1)))

        # Evaluation and learning rate updates.
        run_eval(vqg, val_data_loader, criterion, l2_criterion,
                 args, epoch, scheduler, info_scheduler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Session parameters.
    
    parser.add_argument('--model-path', type=str, default='./weights_with_centerloss/finetuned/',
                        help='Path for saving trained models')
    parser.add_argument('--crop-size', type=int, default=224,
                        help='Size for randomly cropping images')
    parser.add_argument('--log-step', type=int, default=10,
                        help='Step size for prining log info')
    parser.add_argument('--save-step', type=int, default=None,
                        help='Step size for saving trained models')
    parser.add_argument('--eval-steps', type=int, default=100,
                        help='Number of eval steps to run.')
    parser.add_argument('--eval-every-n-steps', type=int, default=300,
                        help='Run eval after every N steps.')
    parser.add_argument('--num-epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--info-learning-rate', type=float, default=0.01)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--max-examples', type=int, default=None,
                        help='For debugging. Limit examples in database.')

    # Lambda values.
    parser.add_argument('--lambda_reg', type=float, default=2.0,
                        help='coefficient to be added in front of the regularisation loss.')
    parser.add_argument('--lambda_gen', type=float, default=2.0,
                        help='coefficient to be added in front of the generation loss.')
    parser.add_argument('--lambda_t', type=float, default=3.0,
                        help='coefficient to be added with the category type space loss.')
    parser.add_argument('--lambda_c', type=float, default=2.0,
                        help='coefficient to be added with the category recon loss.')
    parser.add_argument('--lambda_c_cycle', type=float, default=2.0,
                        help='coefficient to be added with the category cycle loss.')
    parser.add_argument('--lambda_i', type=float, default=1,
                        help='coefficient to be added with the image recon loss.')
    parser.add_argument('--lambda_centerloss', type=float, default=2.0,
                        help='coefficient for center loss')
    
    # Data parameters.
    parser.add_argument('--vocab-path', type=str,
                        default='data/processed/vocab_iq.json',
                        help='Path for vocabulary wrapper.')
    parser.add_argument('--dataset', type=str,
                        default='data/processed/iq_dataset.hdf5',
                        help='Path for train annotation json file.')
    parser.add_argument('--val-dataset', type=str,
                        default='data/processed/iq_val_dataset.hdf5',
                        help='Path for train annotation json file.')
    parser.add_argument('--train-dataset-weights', type=str,
                        default='data/processed/iq_train_dataset_weights.json',
                        help='Location of sampling weights for training set.')
    parser.add_argument('--val-dataset-weights', type=str,
                        default='data/processed/iq_val_dataset_weights.json',
                        help='Location of sampling weights for training set.')
    parser.add_argument('--cat2name', type=str,
                        default='data/processed/cat2name.json',
                        help='Location of mapping from category to type name.')
    parser.add_argument('--load-model', type=str, default=None,
                        help='Location of where the model weights are.')


    # Model parameters
    parser.add_argument('--rnn_cell', type=str, default='LSTM',
                        help='Type of rnn cell (GRU, RNN or LSTM).')
    parser.add_argument('--hidden_size', type=int, default=256,           # changed from 512
                        help='Dimension of lstm hidden states.')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of layers in lstm.')
    parser.add_argument('--max_length', type=int, default=20,
                        help='Maximum sequence length for outputs.')
    parser.add_argument('--encoder_max_len', type=int, default=8,
                        help='Maximum sequence length for inputs.')
    parser.add_argument('--bidirectional', action='store_true', default=False,
                        help='Boolean whether the RNN is bidirectional.')
    parser.add_argument('--use_glove', action='store_true',
                        help='Whether to use GloVe embeddings.')
    parser.add_argument('--embedding_name', type=str, default='6B',
                        help='Name of the GloVe embedding to use.')
    parser.add_argument('--num_categories', type=int, default=16,
                        help='Number of answer types we use.')
    parser.add_argument('--dropout_p', type=float, default=0.2,
                        help='Dropout applied to the RNN model.')
    parser.add_argument('--input_dropout_p', type=float, default=0.2,
                        help='Dropout applied to inputs of the RNN.')
    parser.add_argument('--num_att_layers', type=int, default=2,
                        help='Number of attention layers.')
    parser.add_argument('--z_size', type=int, default=64,
                        help='Dimensions to use for hidden variational space.')
    parser.add_argument('--z_img', type=int, default=256,
                        help='Dimensions to use for encoded images.')
    parser.add_argument('--z_category', type=int, default=4,
                        help='Dimensions to use for encoded categories.')

    # Ablations.
    parser.add_argument('--no-image-recon', action='store_true', default=False,
                        help='Does not try to reconstruct image.')
    parser.add_argument('--no-category-space', action='store_true', default=False,
                        help='Does not try to reconstruct category.')
    parser.add_argument('--step_one', type=bool, default=True,
                        help='For implementing step one of our approach.')
    parser.add_argument('--step_two', type=bool, default=True,
                        help='For implementing step two of our approach.')
    parser.add_argument('--center_loss', type=bool, default=True,
                        help='For implementing center loss.')
    parser.add_argument('--bayes', type=bool, default=True,
                        help='For adding a bayes-vae prior.')

    parser.add_argument('--train_log_file_suffix',default='train',type=str)

    args = parser.parse_args()
    
    train(args)

    # Hack to disable errors for importing Vocabulary. Ignore this line.
    Vocabulary()
