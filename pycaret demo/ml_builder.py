class MLBuilder:
    def __init__(self, model, data):
        self.model = model
        self.data = data

    def build(self):
        self.model.fit(self.data)
        return