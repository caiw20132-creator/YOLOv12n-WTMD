
import torch.nn as nn

from ultralytics.nn.modules.sf1_sppf import SF1SPPF


class MaxPoolSF1SPPF(SF1SPPF):
    """
    Strict pool-type control for SF1SPPF.

    Preserved:
      - argument parsing;
      - delayed layer construction;
      - optional adapter;
      - cv1 and cv2;
      - hidden-channel rule;
      - three serial pooling calls;
      - concatenation order.

    Changed:
      SoftPool2d -> nn.MaxPool2d only.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pool = nn.MaxPool2d(
            kernel_size=self.k,
            stride=1,
            padding=self.k // 2,
        )


def register_maxpool_sf1sppf():
    import ultralytics.nn.tasks as tasks
    import ultralytics.nn.modules as modules

    tasks.__dict__["MaxPoolSF1SPPF"] = MaxPoolSF1SPPF
    modules.__dict__["MaxPoolSF1SPPF"] = MaxPoolSF1SPPF

    setattr(tasks, "MaxPoolSF1SPPF", MaxPoolSF1SPPF)
    setattr(modules, "MaxPoolSF1SPPF", MaxPoolSF1SPPF)

    if hasattr(modules, "__all__") and isinstance(modules.__all__, list):
        if "MaxPoolSF1SPPF" not in modules.__all__:
            modules.__all__.append("MaxPoolSF1SPPF")
