import numpy as np
from lib.Encoding.EncodingEnum import Strategy
from lib.Encoding.EncodingConfig import CustomEncoding


# Author: Daniel Vera morales
class EncodeHelper:

    def __init__(self, n_classes=1, encodingStrategy=None):
        self.encodingStrategy = encodingStrategy
        self.encoded_Y = None
        self.index_to_value = None
        self.value_to_index = None
        self.n_classes = n_classes

    def one_hot_encoding(self, Y):
        if self.encodingStrategy is None:
            self.encodingStrategy = CustomEncoding(None, None, Strategy.ENCODE_BY_CLASSES)
        if self.encodingStrategy.strategy == Strategy.ENCODE_BY_CLASSES:
            return self.one_hot_encoding_by_classes(Y)
        elif self.encodingStrategy.strategy == Strategy.REPLACE_LABEL_BY_VALUE:
            return self.one_hot_encoding_custom_value(Y, self.encodingStrategy)
        elif self.encodingStrategy.strategy == Strategy.TAKE_LABEL_AS_INDEX:
            return self.one_hot_encoding_label_as_index(Y)
        elif self.encodingStrategy.strategy == Strategy.CUSTOM_MAPPING:
            if self.encodingStrategy.prepared_data is None:
                raise ValueError("CUSTOM_MAPPING strategy needs prepared_data field not None")
            else:
                self.one_hot_encoding_mapping(Y)

    def one_hot_encoding_by_classes(self, Y):
        n_labels = Y.shape[0]
        self.encoded_Y = np.zeros((n_labels, self.n_classes))
        self.value_to_index, self.index_to_value = self.encode_values(Y)
        for i in range(n_labels):
            label = Y[i]
            label = self.value_to_index.get(label)
            self.encoded_Y[i][label] = 1
        return self.encoded_Y

    def one_hot_encoding_label_as_index(self, Y):
        n_labels = Y.shape[0]
        self.encoded_Y = np.zeros((n_labels, self.n_classes))
        for i in range(n_labels):
            label = Y[i]
            self.encoded_Y[i][label] = 1
        return self.encoded_Y

    def one_hot_encoding_custom_value(self, Y, strategy):
        n_labels = Y.shape[0]
        self.encoded_Y = np.zeros((n_labels, 1))
        for i in range(n_labels):
            self.encoded_Y[i] = strategy.encode_cb(Y[i])
        return self.encoded_Y

    def one_hot_encoding_mapping(self, Y):
        n_labels = Y.shape[0]
        self.encoded_Y = np.zeros((n_labels, 1))
        for i in range(n_labels):
            label = Y[i]
            label = self.encodingStrategy.prepared_data.get("ktov").get(label)
            self.encoded_Y[i] = label
        return self.encoded_Y

    def decode_value(self, yhat):
        if self.encodingStrategy.strategy == Strategy.REPLACE_LABEL_BY_VALUE:
            return self.one_hot_decoding_custom_value(yhat)

    def one_hot_decoding_custom_value(self, yhat):
        classified_main = []
        for v in yhat:
            classified_tmp = []
            for vv in v:
                classified_tmp.append(self.encodingStrategy.decode_cb(vv))
            classified_main.append(classified_tmp)
        return np.array(classified_main)

    def value_from_index(self, index):
        return self.index_to_value.get(index)

    def encode_values(self, Y):
        unique_values = np.unique(Y)
        value_to_index = {v: ord(v) % len(unique_values) for v in unique_values}
        index_to_value = {v: k for k, v in value_to_index.items()}
        return value_to_index, index_to_value
