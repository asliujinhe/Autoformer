import logging
import os

import numpy as np
import torch

from data_provider.data_factory import data_provider
from models import Informer, Autoformer, Transformer, Reformer

# Set up logging configuration
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S', level=logging.INFO)


class ExpPredict:
    """
    A class for loading the model and making predictions.

    Args:
        args (object): A configuration object containing the model parameters.
    """

    def __init__(self, args):
        self.args = args

        self.setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des, 0)
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.args.use_gpu else "cpu")

        # Build the model
        self.model = self._build_model()
        self.model.to(self.device)

    def _build_model(self):
        """
        Build the model based on the given configuration.
        """
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'Reformer': Reformer,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = torch.nn.DataParallel(model, device_ids=self.args.device_ids)

        return model

    def _get_data(self, dataframe):
        """
        Get the data loader for the given flag (train, validation, test, pred).
        """
        self.args.dataframe = dataframe
        data_set, data_loader = data_provider(self.args, 'pred')
        return data_set, data_loader

    def _predict(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        """
        Perform prediction using the model.
        """
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

        def _run_model():
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            if self.args.output_attention:
                outputs = outputs[0]
            return outputs

        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = _run_model()
        else:
            outputs = _run_model()

        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

        return outputs, batch_y

    def predict(self, dataframe, load=True):
        """
        Run the prediction task using the provided setting.

        dataframe must consist of the following columns:
        - 'date': The date of the data.
        - `target`: The target variable. Depends on your arguments on `target`.

        Args:
            dataframe (pandas.DataFrame): The dataframe containing the data.
            load (bool): Whether to load the best model from checkpoints.
        """
        # Load prediction data
        _, pred_loader = self._get_data(dataframe)

        # Load the model if required
        if load:
            path = os.path.join(self.args.checkpoints, self.setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            logging.info(f'Loading model from {best_model_path}')
            self.model.load_state_dict(torch.load(best_model_path))

        predicts = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Perform prediction
                outputs, batch_y = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)

                pred = outputs.detach().cpu().numpy()
                predicts.append(pred)

        # Concatenate predictions
        predicts = np.array(predicts)
        predicts = predicts.reshape(-1, predicts.shape[-2], predicts.shape[-1])
        return predicts