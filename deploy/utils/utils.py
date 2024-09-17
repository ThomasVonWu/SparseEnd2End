import numpy as np


def printArrayInformation(x: np.ndarray, logger, info: str, prefix: str):
    logger.debug(f"{prefix}: [{info}]")
    logger.debug(
        "\tMax=%.3f, Min=%.3f, SumAbs=%.3f"
        % (
            np.max(x),
            np.min(x),
            np.sum(abs(x)),
        )
    )
    logger.debug(
        "\tfirst5 | last5 %s  ......  %s" % (x.reshape(-1)[:5], x.reshape(-1)[-5:])
    )
