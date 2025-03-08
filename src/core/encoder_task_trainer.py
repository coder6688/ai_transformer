from core.accuracy_calculation import calculate_accuracy
from core.transformer_encoder_trainer import TransformerEncoderTrainer


class EncoderTaskTrainer(TransformerEncoderTrainer):
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
        inp_data, labels = batch
        
        # If convert to one-hot/embeded vector in inp_data: (batch_size, seq_len, num_classes)
        # eg.   
        # 1. Takes input tensor of shape [batch_size, seq_len] with integer class indices (0-9)
        # 2. Converts to [batch_size, seq_len, num_classes] tensor
        # F.one_hot() example:
        # Input (class indices) for batch=1: [[3, 0, 4]]
        # Output (one-hot) for batch=1:
        # [[[0,0,0,1,0,0,0,0,0,0],
        #   [1,0,0,0,0,0,0,0,0,0],
        #   [0,0,0,0,1,0,0,0,0,0]]]

        # Customized embedding function passed in
        inp_data = self.embed_layer(inp_data)
        
        #  Perform prediction and calculate loss and accuracy
        preds = self.forward(inp_data, need_pos_encoding=True)

        if not hasattr(self.hparams, "is_binary_classification"):
            self.hparams.is_binary_classification = False

        if self.hparams.is_binary_classification:
            # in transformer_predictor already applies sigmoid
            # TODO: how to ignore those padded words
            loss = self.criterion(preds, labels)

            # Convert probabilities to predicted class (eg. 0 or 1 from imdb_dataset)
            preds_label = (preds >= 0.5).int()
            
            # Calculate accuracy
            correct = (preds_label == labels).float()
            acc = correct.sum() / len(labels)

        else:
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
