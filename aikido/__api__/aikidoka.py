import torch


class Aikidoka(torch.nn.Module):

    def save(self, path: str):
        self.eval()
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.eval()
        self.load_state_dict(torch.load(path))

    def to_onnx(self, path: str = "./model.onnx", opset_version: int = 14):
        with torch.no_grad():
            self.eval()
            torch.onnx.export(
                self,
                self.info.model_args,
                f=path,
                # input_names=self.info.input_names,
                # output_names=self.info.output_names,
                # dynamic_axes=self.info.dynamic_axes,
                do_constant_folding=False,
                opset_version=opset_version
            )
