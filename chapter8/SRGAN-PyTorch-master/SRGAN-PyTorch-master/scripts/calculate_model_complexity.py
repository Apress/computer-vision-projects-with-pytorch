import torch
from time import perf_counter
import logging

from thop import profile
from srgan_pytorch.models import srgan

logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)

IMG_SIZE = 96
IMG_CHS = 3
BATCH_SIZE = 64

def main():
    data = torch.randn([1, IMG_CHS, IMG_SIZE, IMG_SIZE])
    model = srgan(pretrained=False)
    model.eval()

    params = sum(x.numel() for x in model.parameters()) / 1E6
    flops_calc = profile(model=model, inputs=(data,), verbose=False)[0] / 1E9 * 2



if __name__ == "__main__":
    logger.info("ScriptEngine:")
    logger.info("\tAPI version .......... 0.3.0")
    logger.info("\tBuild ................ 2021.07.02")

    main()
