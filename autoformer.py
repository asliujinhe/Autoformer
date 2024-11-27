import torch

from exp.exptrain import ExpTrain
from exp.exppredict import ExpPredict


class Autoformer:
    """
    A class to encapsulate the time series forecasting model and prediction workflow.
    """

    def __init__(self, model_id='Test_96_96', model='Autoformer', data='custom', root_path='./dataset/test/',
                 data_path='test.csv',
                 features='MS', target='corn_close', freq='h', checkpoints='./checkpoints/', seq_len=96, label_len=48,
                 pred_len=96, bucket_size=4, n_hashes=4, enc_in=11, dec_in=11, c_out=11, d_model=512, n_heads=8,
                 e_layers=2, d_layers=1, d_ff=2048, moving_avg=25, factor=3, distil=True, dropout=0.05, embed='learned',
                 activation='gelu', output_attention=False, num_workers=10, itr=1, use_gpu=True, gpu=0,
                 use_multi_gpu=False, devices='0,1,2,3', patience=5, train_epochs=10, batch_size=32,
                 learning_rate=0.0001, loss='mse', lradj='type1', des='test', use_amp=False):
        """
        初始化模型配置参数。

        参数：
        ----------
        model_id : str, default 'Test_96_96'
            模型的唯一标识符，用于保存和加载模型。

        model : str, default 'Autoformer'
            模型名称，支持的选项包括 'Autoformer', 'Informer', 'Transformer'。

        data : str, default 'custom'
            数据集类型，用于选择训练数据集。

        root_path : str, default './dataset/test/'
            数据文件的根路径，所有数据文件的相对路径从此路径开始。

        data_path : str, default 'test.csv'
            数据文件路径，包含要加载的训练数据。

        features : str, default 'MS'
            特征类型，支持的选项包括：
            - 'M'：多变量预测多变量
            - 'S'：单变量预测单变量
            - 'MS'：多变量预测单变量

        target : str, default 'corn_close'
            目标特征，仅在 S 或 MS 类型的任务中使用。

        freq : str, default 'h'
            时间特征编码的频率，支持的选项包括：
            - 's'：秒
            - 't'：分钟
            - 'h'：小时
            - 'd'：天
            - 'b'：工作日
            - 'w'：周
            - 'm'：月
            或更精确的频率（例如 15min 或 3h）。

        checkpoints : str, default './checkpoints/'
            模型检查点的保存路径。

        seq_len : int, default 96
            输入序列的长度。

        label_len : int, default 48
            标签序列的长度（即起始令牌的长度）。

        pred_len : int, default 96
            预测序列的长度。

        bucket_size : int, default 4
            用于 Reformer 模型的 bucket size。

        n_hashes : int, default 4
            用于 Reformer 模型的 hash 数量。

        enc_in : int, default 11
            编码器输入的大小。

        dec_in : int, default 11
            解码器输入的大小。

        c_out : int, default 11
            输出大小。

        d_model : int, default 512
            模型的维度。

        n_heads : int, default 8
            多头注意力机制中的头数。

        e_layers : int, default 2
            编码器层数。

        d_layers : int, default 1
            解码器层数。

        d_ff : int, default 2048
            前馈网络的维度。

        moving_avg : int, default 25
            移动平均窗口大小。

        factor : int, default 3
            注意力因子，决定注意力机制的规模。

        distil : bool, default True
            是否在编码器中使用蒸馏机制。`False` 表示不使用。

        dropout : float, default 0.05
            Dropout 的比例，防止过拟合。

        embed : str, default 'learned'
            时间特征编码方式，支持的选项包括：
            - 'timeF'：使用时间特征进行编码
            - 'fixed'：固定编码
            - 'learned'：学习型编码

        activation : str, default 'gelu'
            激活函数，支持的选项包括：
            - 'gelu'
            - 'relu'
            - 'tanh'

        output_attention : bool, default False
            是否输出注意力层的输出。

        num_workers : int, default 10
            数据加载器的工作线程数量。

        itr : int, default 1
            实验次数，用于执行多次实验的平均结果。

        use_gpu : bool, default True
            是否使用 GPU 进行训练。

        gpu : int, default 0
            使用的 GPU 编号。

        use_multi_gpu : bool, default False
            是否使用多个 GPU 进行训练。

        devices : str, default '0,1,2,3'
            多个 GPU 的设备编号（如果使用多个 GPU）。

        patience : int, default 5
            提前停止的耐心度，即多少轮没有改进时停止训练。

        train_epochs : int, default 10
            训练的总轮数。

        batch_size : int, default 32
            每批次的训练样本数。

        learning_rate : float, default 0.0001
            优化器的学习率。

        loss : str, default 'mse'
            损失函数的类型，常见的选择包括 'mse'（均方误差）和 'mae'（绝对误差）。

        lradj : str, default 'type1'
            学习率调整方式，支持不同类型的调整策略。

        des : str, default 'test'
            实验描述，用于记录实验的说明。

        use_amp : bool, default False
            是否使用自动混合精度（AMP）进行训练，以加速训练过程并减少内存使用。
        """
        self.model_id = model_id
        self.model = model
        self.data = data
        self.root_path = root_path
        self.data_path = data_path
        self.features = features
        self.target = target
        self.freq = freq
        self.checkpoints = checkpoints
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.bucket_size = bucket_size
        self.n_hashes = n_hashes
        self.enc_in = enc_in
        self.dec_in = dec_in
        self.c_out = c_out
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.d_ff = d_ff
        self.moving_avg = moving_avg
        self.factor = factor
        self.distil = distil
        self.dropout = dropout
        self.embed = embed
        self.activation = activation
        self.output_attention = output_attention
        self.num_workers = num_workers
        self.itr = itr
        self.use_gpu = use_gpu
        self.gpu = gpu
        self.use_multi_gpu = use_multi_gpu
        self.devices = devices
        self.patience = patience
        self.train_epochs = train_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.loss = loss
        self.lradj = lradj
        self.des = des
        self.use_amp = use_amp


        # Set GPU usage based on available devices
        self.use_gpu = True if torch.cuda.is_available() and self.use_gpu else False
        if self.use_gpu and self.use_multi_gpu:
            self.devices = self.devices.replace(' ', '')
            device_ids = self.devices.split(',')
            self.device_ids = [int(id_) for id_ in device_ids]
            self.gpu = self.device_ids[0]

        self.exp_train = ExpTrain(self)
        self.exp_predict = ExpPredict(self)

    def train(self, dataframe=None):
        """
        Train the model on the given dataset.

        Args:
            dataframe (pandas.DataFrame): The dataset to train the model on.
        """
        self.exp_train.train(dataframe)

    def predict(self, dataframe=None):
        """
        Predict the future values of the time series.

        """

        return self.exp_predict.predict(dataframe)

