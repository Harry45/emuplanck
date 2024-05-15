import torch


class yTransformMoped:
    def __init__(self, values):
        self.values = torch.from_numpy(values)
        self.ymean, self.ystd, self.train_y = self.forward_transform(self.values)

    def forward_transform(self, values):
        """
        Implement a forward transformation if we want to.
        """
        ymean = torch.mean(values)
        ystd = torch.std(values)
        train_y = (values - ymean) / ystd
        train_y = train_y.to(torch.float32)
        return ymean, ystd, train_y

    def inverse_tranform(self, prediction):
        """
        Apply the inverse transformation on the predicted values.
        """
        pred_trans = prediction * self.ystd + self.ymean
        return pred_trans
