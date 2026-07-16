"""
Experiment Logger
"""

import logging

logging.basicConfig(

    filename="../results/experiment.log",

    level=logging.INFO,

    format="%(asctime)s %(message)s"

)

logging.info("Experiment Started")