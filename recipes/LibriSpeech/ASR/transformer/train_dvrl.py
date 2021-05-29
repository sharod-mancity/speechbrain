
#!/usr/bin/env python3
"""Recipe for training a Transformer ASR system with librispeech.
The system employs an encoder, a decoder, and an attention mechanism
between them. Decoding is performed with (CTC/Att joint) beamsearch coupled with a neural
language model.
To run this recipe, do the following:
> python train.py hparams/transformer.yaml
> python train.py hparams/conformer.yaml
With the default hyperparameters, the system employs a convolutional frontend and a transformer.
The decoder is based on a Transformer decoder. Beamsearch coupled with a Transformer
language model is used  on the top of decoder probabilities.
The neural network is trained on both CTC and negative-log likelihood
targets and sub-word units estimated with Byte Pairwise Encoding (BPE)
are used as basic recognition tokens. Training is performed on the full
LibriSpeech dataset (960 h).
The best model is the avergage of the checkpoints from last 5 epochs.
The experiment file is flexible enough to support a large variety of
different systems. By properly changing the parameter files, you can try
different encoders, decoders, tokens (e.g, characters instead of BPE),
training split (e.g, train-clean 100 rather than the full one), and many
other possible variations.
Authors
 * Jianyuan Zhong 2020
 * Mirco Ravanelli 2020
 * Peter Plantinga 2020
 * Samuele Cornell 2020
 * Titouan Parcollet 2021
"""
import os
import sys
import time
from tqdm.contrib import tqdm
import torch
import logging
import copy
from pathlib import Path
import speechbrain as sb
import numpy as np
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
from torch.utils.data import DataLoader

from speechbrain.nnet.mlp import MLP, LayerNorm


logger = logging.getLogger(__name__)


# Define training procedure
class ASR(sb.core.Brain):

    def data_value_estimator(self,y_pred_diff,y_input):
        inp_dim = 80
        #code to add a neural network for data value estimator.
        mlp_options={}
        mlp_options['dnn_lay']='80,50,1'
        mlp_options['dnn_drop']='0.15,0.0,0.0'
        mlp_options['dnn_use_batchnorm']='False,False,False'
        mlp_options['dnn_use_laynorm']='True,False,False'
        mlp_options['dnn_use_laynorm_inp']='True'
        mlp_options['dnn_use_batchnorm_inp']='False'
        mlp_options['dnn_act']='relu,relu,sigmoid'

        return MLP(mlp_options, inp_dim)

    def fit_dvrl(
        self,
        epoch_counter,
        train_set,
        valid_set=None,
        progressbar=None,
        train_loader_kwargs={},
        valid_loader_kwargs={},
    ):
        """Iterate epochs and datasets to improve objective with data valuation.
        Data Valuation using Reinforcement Learning - https://arxiv.org/abs/1909.11671
        Relies on the existence of multiple functions that can (or should) be
        overridden. The following methods are used and expected to have a
        certain behavior:
        * ``fit_batch()``
        * ``evaluate_batch()``
        * ``update_average()``
        * ``data_value_estimator()``
        If the initialization was done with distributed_count > 0 and the
        distributed_backend is ddp, this will generally handle multiprocess
        logic, like splitting the training data into subsets for each device and
        only saving a checkpoint on the main process.
        Arguments
        ---------
        epoch_counter : iterable
            Each call should return an integer indicating the epoch count.
        train_set : Dataset, DataLoader
            A set of data to use for training. If a Dataset is given, a
            DataLoader is automatically created. If a DataLoader is given, it is
            used directly.
        valid_set : Dataset, DataLoader
            A set of data to use for validation. If a Dataset is given, a
            DataLoader is automatically created. If a DataLoader is given, it is
            used directly.
        train_loader_kwargs : dict
            Kwargs passed to `make_dataloader()` for making the train_loader
            (if train_set is a Dataset, not DataLoader).
            E.G. batch_size, num_workers.
            DataLoader kwargs are all valid.
        valid_loader_kwargs : dict
            Kwargs passed to `make_dataloader()` for making the valid_loader
            (if valid_set is a Dataset, not DataLoader).
            E.g., batch_size, num_workers.
            DataLoader kwargs are all valid.
        progressbar : bool
            Whether to display the progress of each epoch in a progressbar.
        """

        if not isinstance(train_set, DataLoader):
            train_set = self.make_dataloader(
                train_set, stage=sb.Stage.TRAIN, **train_loader_kwargs
            )
        if valid_set is not None and not isinstance(valid_set, DataLoader):
            valid_set = self.make_dataloader(
                valid_set,
                stage=sb.Stage.VALID,
                ckpt_prefix=None,
                **valid_loader_kwargs,
            )

        #get DVRL data value estimator:
        data_value_estimator = self.data_value_estimator().to(self.device)
        #create its optimizer
        dve_optimizer = torch.optim.SGD([{'params': data_value_estimator.parameters()}], \
                                            lr=0.01, momentum=0.9, \
											weight_decay=5e-4, nesterov=True)

        #reward
        reward = 0
        self.epsilon = 1e-8  # Adds to the log to avoid overflow
        self.threshold = 0.9  # Encourages exploration

        valid_perf = 18.54 #static from the previous runs.
        
        if progressbar is None:
            progressbar = not self.noprogressbar

        # Training stage
        self.modules.train()

        # Reset nonfinite count to 0 each epoch
        self.nonfinite_count = 0

        # Time since last intra-epoch checkpoint
        last_ckpt_time = time.time()

        inner_epochs = 20

        pred_model = self.modules
        
		predictions = self.compute_forward(train_set, feats, sb.Stage.VALID)
        # Only show progressbar if requested and main_process
        enable = progressbar and sb.utils.distributed.if_main_process()
        with tqdm(train_set, initial=self.step, dynamic_ncols=True, disable=not enable,) as t:
            
            #processing for each batch.
            for batch in t:
                logger.debug("Processing New Batch")
                #initialize the training module for each batch.
                self.optimizer = self.hparams.SGD(self.modules.parameters())
                self.modules = copy.deepcopy(pred_model)
                self.modules.to(self.device)
                self.modules.train()

                batch = batch.to(self.device)

                # compute features
                wavs, wav_lens = batch.sig
                feats = self.hparams.compute_features(wavs)
                current_epoch = self.hparams.epoch_counter.current
                feats = self.hparams.normalize(feats, wav_lens, epoch=current_epoch)
                tokens_eos, tokens_eos_lens = batch.tokens_eos
                _,y_train_valid_pred,_,_=self.compute_forward(batch, feats, sb.Stage.TRAIN)
                y_pred_diff = np.abs(tokens_eos - y_train_valid_pred)
                for i in range(inner_epochs):
                    #For each batch run it for inner epochs

                    #(1) get selection probabilities from the data value estimator.
                    selection_probabilities = data_value_estimator(feats,y_pred_diff,tokens_eos)
                
                    #(2) Binomial sampling
                    sp = selection_probabilities.cpu().clone().detach().numpy() #create a clone and then get the numpy array
                    current_selection_probabilities = np.random.binomial(1, sp, sp.shape)
                    # Exception (When selection probability is 0)
                    if np.sum(current_selection_probabilities) == 0:
                        selection_probabilities = 0.5 * np.ones(np.shape(current_selection_probabilities))
                        current_selection_probabilities = np.random.binomial(1, selection_probabilities, selection_probabilities.shape)
                    
                    current_selection_probabilities = torch.from_numpy(current_selection_probabilities).to(self.device) #move np array to torch
                    #(3) New_batch creation
                    sample_weights = current_selection_probabilities[:, 0]


                    #(4) Main objective loss optimization
                    for i in range(10):
                      train_loss = self.fit_batch(batch, feats, sample_weights)

                    # Validation stage
                    if valid_set is not None:
                        self.modules.eval()
                        avg_valid_loss = 0.0
                        self.step = 0
                        with torch.no_grad():
                            for batch in valid_set:

                                batch = batch.to(self.device)
                                #feature computation for the validation set.
                                wavs, wav_lens = batch.sig
                                feats = self.hparams.compute_features(wavs)
                                current_epoch = self.hparams.epoch_counter.current
                                feats = self.hparams.normalize(feats, wav_lens, epoch=current_epoch)

                                self.step += 1
                                loss = self.evaluate_batch(batch, feats, stage=sb.Stage.VALID)
                                avg_valid_loss += loss.detach()

                                if self.step == 10:
                                    break #stop after 10 batches.

                        avg_valid_loss = avg_valid_loss / self.step  
                    
                    #compute dve loss
                    reward = avg_valid_loss - valid_perf #(difference of current performance vs gold performance)
                    
                    #(5) DVRL loss optimization
                    # Generator loss (REINFORCE algorithm)
                    prob = torch.sum(current_selection_probabilities * torch.log(selection_probabilities + self.epsilon) +\
                                    (1-current_selection_probabilities) * torch.log(1 - selection_probabilities + self.epsilon))

                    a, _ = torch.max(torch.mean(selection_probabilities) - self.threshold, 0)
                    b, _ = torch.max((1-self.threshold) - torch.mean(selection_probabilities), 0)
                    dve_loss = (-reward * prob) + (a + b) * 1e3
                    
                    logger.debug("Train Loss: {}".format(train_loss))
                    logger.debug("reward: {}".format(reward))
                    logger.debug("Dve Loss: {}".format(dve_loss))
                    dve_loss.backward()

                    dve_optimizer.step()
                    dve_optimizer.zero_grad()

        torch.save({'dve': data_value_estimator.state_dict()},
							os.path.join(hparams["output_folder"], 'dve.ckpt'))

    def compute_forward(self, batch, feats, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_bos, _ = batch.tokens_bos

        # Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            if hasattr(self.modules, "env_corrupt"):
                wavs_noise = self.modules.env_corrupt(wavs, wav_lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                wav_lens = torch.cat([wav_lens, wav_lens])
                tokens_bos = torch.cat([tokens_bos, tokens_bos], dim=0)

        #compute features (feats are computed already for dve step, so passing as an argument here.)
        #feats = self.hparams.compute_features(wavs)
        #current_epoch = self.hparams.epoch_counter.current
        #feats = self.hparams.normalize(feats, wav_lens, epoch=current_epoch)

        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                feats = self.hparams.augmentation(feats)

        # forward modules
        src = self.hparams.CNN(feats)
        enc_out, pred = self.hparams.Transformer(
            src, tokens_bos, wav_lens, pad_idx=self.hparams.pad_index
        )

        # output layer for ctc log-probabilities
        logits = self.hparams.ctc_lin(enc_out)
        p_ctc = self.hparams.log_softmax(logits)

        # output layer for seq2seq log-probabilities
        pred = self.hparams.seq_lin(pred)
        p_seq = self.hparams.log_softmax(pred)

        # Compute outputs
        hyps = None
        if stage == sb.Stage.TRAIN:
            hyps = None
        elif stage == sb.Stage.VALID:
            hyps = None
            current_epoch = self.hparams.epoch_counter.current
            if current_epoch % self.hparams.valid_search_interval == 0:
                # for the sake of efficeincy, we only perform beamsearch with limited capacity
                # and no LM to give user some idea of how the AM is doing
                hyps, _ = self.hparams.valid_search(enc_out.detach(), wav_lens)
        elif stage == sb.Stage.TEST:
            hyps, _ = self.hparams.test_search(enc_out.detach(), wav_lens)

        return p_ctc, p_seq, wav_lens, hyps

    def compute_objectives(self, predictions, batch, stage, sample_weights=None):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        (p_ctc, p_seq, wav_lens, hyps,) = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        if hasattr(self.modules, "env_corrupt") and stage == sb.Stage.TRAIN:
            tokens_eos = torch.cat([tokens_eos, tokens_eos], dim=0)
            tokens_eos_lens = torch.cat(
                [tokens_eos_lens, tokens_eos_lens], dim=0
            )
            tokens = torch.cat([tokens, tokens], dim=0)
            tokens_lens = torch.cat([tokens_lens, tokens_lens], dim=0)

        if not torch.is_tensor(sample_weights):
            loss_seq = self.hparams.seq_cost(p_seq, tokens_eos, length=tokens_eos_lens)
            loss_ctc = self.hparams.ctc_cost(p_ctc, tokens, wav_lens, tokens_lens)
            loss = (
                self.hparams.ctc_weight * loss_ctc
                + (1 - self.hparams.ctc_weight) * loss_seq
            )
        else:
            #sample weights are present, compute the weighted loss w.r.t  the given weights.
            loss_seq = self.hparams.seq_cost(p_seq, tokens_eos, length=tokens_eos_lens, reduction="batch")
            loss_ctc = self.hparams.ctc_cost(p_ctc, tokens, wav_lens, tokens_lens, reduction="batch")

            loss = (self.hparams.ctc_weight * loss_ctc + (1 - self.hparams.ctc_weight) * loss_seq)

            loss = loss * sample_weights #weight the loss for each sample.
            loss = torch.sum(loss) / self.hparams.batch_size #average the loss across the batch.

        return loss

    def fit_batch(self, batch, feats, sample_weights):
        """Train the parameters given a single batch in input"""
        # check if we need to switch optimizer
        # if so change the optimizer from Adam to SGD
        #self.check_and_reset_optimizer()

        predictions = self.compute_forward(batch, feats, sb.Stage.TRAIN)
        
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN, sample_weights=sample_weights)

        # normalize the loss by gradient_accumulation step
        (loss / self.hparams.gradient_accumulation).backward()

        # gradient clipping & early stop if loss is not fini
        self.check_gradients(loss)

        self.optimizer.step()
        self.optimizer.zero_grad()

        # anneal lr every update
        self.hparams.noam_annealing(self.optimizer)

        return loss.detach()

    def evaluate_batch(self, batch, feats, stage):
        """Computations needed for validation/test batches"""
        with torch.no_grad():
            predictions = self.compute_forward(batch, feats, stage=stage)
            loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    # test is separate
    test_datasets = {}
    for csv_file in hparams["test_csv"]:
        name = Path(csv_file).stem
        test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_file, replacements={"data_root": data_folder}
        )
        test_datasets[name] = test_datasets[name].filtered_sorted(
            sort_key="duration"
        )

    datasets = [train_data, valid_data] + [i for k, i in test_datasets.items()]

    # We get the tokenizer as we need it to encode the labels when creating
    # mini-batches.
    tokenizer = hparams["tokenizer"]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        tokens_list = tokenizer.encode_as_ids(wrd)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "wrd", "tokens_bos", "tokens_eos", "tokens"],
    )
    return train_data, valid_data, test_datasets, tokenizer


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # 1.  # Dataset prep (parsing Librispeech)
    from librispeech_prepare import prepare_librispeech  # noqa

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_librispeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            "tr_splits": hparams["train_splits"],
            "dev_splits": hparams["dev_splits"],
            "te_splits": hparams["test_splits"],
            "save_folder": hparams["data_folder"],
            "merge_lst": hparams["train_splits"],
            "merge_name": hparams["train_csv"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    # here we create the datasets objects as well as tokenization and encoding
    train_data, valid_data, test_datasets, tokenizer = dataio_prepare(hparams)

    # We download the pretrained LM from HuggingFace (or elsewhere depending on
    # the path given in the YAML file). The tokenizer is loaded at the same time.
    run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected(device=run_opts["device"])

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["Adam"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # adding objects to trainer:
    asr_brain.tokenizer = hparams["tokenizer"]

    # Training
    asr_brain.fit_dvrl(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )
