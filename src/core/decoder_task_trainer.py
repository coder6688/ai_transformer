import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch

from core.accuracy_calculation import calculate_accuracy
from core.transformer_decoder_trainer import TransformerDecoderTrainer


class DecoderTaskTrainer(TransformerDecoderTrainer):
    def training_step(self, batch, batch_idx):
        loss, _ = self._calculate_loss(batch, mode="train", batch_idx=batch_idx)
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")

    def _calculate_loss(self, batch, mode="train", batch_idx=0):
        # inp_data: (batch_size, seq_len)   
        # labels: (batch_size, seq_len)
        inp_seq_data, labels = batch
        
        # must create padding mask here and then pass along
        if self.hparams.padding_mask is not None and self.hparams.padding_mask:
            # q and k can have different seq length here 
            not_padding_mask = inp_seq_data != self.hparams.pad_id
            not_q_padding_mask = not_padding_mask.unsqueeze(-1)
            not_k_padding_mask = not_padding_mask.unsqueeze(-2)
            padding_mask = ~ (not_q_padding_mask & not_k_padding_mask)

            # expand in num_of_head direction
            padding_mask = padding_mask.unsqueeze_(1)
        else:
            padding_mask = None

        # Customized embedding function passed in
        inp_data = self.embed_layer(inp_seq_data)
        
        #  Perform prediction and calculate loss and accuracy
        preds = self.forward(inp_data, mask=padding_mask, need_pos_encoding=True)

        loss = self.criterion(preds.view(-1, preds.size(-1)), labels.view(-1))
        pad_id = self.hparams['pad_id'] if 'pad_id' in self.hparams else 0
        acc = calculate_accuracy(preds, labels, pad_id)

        # use CSVLogger to generate metrics.csv 
        # for history plot, also work in tensor board
        if self.training:
            self.log(f"{mode}_loss", loss)
            self.log(f"{mode}_acc", acc)
        else:
            # Log per-batch validation metrics
            self.log(f'{mode}_loss', loss, 
                    on_step=False,    # Log each batch
                    on_epoch=True,  # Also log epoch average
                    prog_bar=False,  # Don't show in progress bar
                    logger=True)     # Send to logger (TensorBoard/WandB/etc)
            self.log(f'{mode}_acc', acc, 
                    on_step=False,    # Log each batch
                    on_epoch=True,  # Also log epoch average
                    prog_bar=False,  # Don't show in progress bar
                    logger=True)     # Send to logger (TensorBoard/WandB/etc)
            
        self.log('lr', self.optimizers().param_groups[0]['lr'], 
                on_step=True, on_epoch=False)
        
        # check if use tensorboard only
        # to view log:
        # eg. tensorboard --logdir ShiftedSeqTask/lightning_logs/version_7
        if hasattr(self.logger.experiment, "add_scalars"):
            task_name = self.hparams["task_name"] if "task_name" in self.hparams else " "
            self.logger.experiment.add_scalars(f'{task_name}', 
                                               {f"{mode}_loss": loss, f"{mode}_acc": acc}, 
                                               batch_idx )

        return loss, acc
