class CustomEncoding:
    def __init__(self, encodeCb, decodeCb, strategy):
        self.prepared_data = None
        self.encode_cb = encodeCb
        self.decode_cb = decodeCb
        self.strategy = strategy

    def prepare_from_custom_map(self, k, v):
        prepared_data = {"ktov": {}, "vtok": {}}
        for i in range(len(k)):
            label = k[i]
            value = v[i]
            prepared_data["ktov"][label] = value
            prepared_data["vtok"][value] = label

        self.prepared_data = prepared_data
        return prepared_data
