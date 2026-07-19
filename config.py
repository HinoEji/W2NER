import json
import os


class Config:
    def __init__(self, args):
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)

        self.dataset = config["dataset"]
        self.save_path = config["save_path"]
        self.predict_path = config["predict_path"]
        self.tokenizer_path = config.get("tokenizer_path")

        self.dist_emb_size = config["dist_emb_size"]
        self.type_emb_size = config["type_emb_size"]
        self.lstm_hid_size = config["lstm_hid_size"]
        self.conv_hid_size = config["conv_hid_size"]
        self.bert_hid_size = config["bert_hid_size"]
        self.biaffine_size = config["biaffine_size"]
        self.ffnn_hid_size = config["ffnn_hid_size"]

        self.dilation = config["dilation"]

        self.emb_dropout = config["emb_dropout"]
        self.conv_dropout = config["conv_dropout"]
        self.out_dropout = config["out_dropout"]

        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]
        self.num_workers = config.get("num_workers", 4)

        self.learning_rate = config["learning_rate"]
        self.weight_decay = config["weight_decay"]
        self.clip_grad_norm = config["clip_grad_norm"]
        self.bert_name = config["bert_name"]
        self.bert_learning_rate = config["bert_learning_rate"]
        self.warm_factor = config["warm_factor"]

        self.use_bert_last_4_layers = config["use_bert_last_4_layers"]

        self.seed = config.get("seed", 123)

        for k, v in args.__dict__.items():
            if v is not None:
                self.__dict__[k] = v
        if self.tokenizer_path is None:
            self.tokenizer_path = self._default_tokenizer_path(self.save_path)

    def __repr__(self):
        return "{}".format(self.__dict__.items())

    @staticmethod
    def _default_tokenizer_path(save_path):
        root, _ = os.path.splitext(save_path)
        return "{}_tokenizer".format(root)
