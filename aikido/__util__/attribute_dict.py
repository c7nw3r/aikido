class AttributeDict(dict):
    def __init__(self, attributes: dict):
        super().__init__()
        for k, v in attributes.items():
            self[k] = v

    def __getattr__(self, item):
        return self[item]
