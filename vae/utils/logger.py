import sys
import logging
import os

logging.getLogger("requests").setLevel(logging.CRITICAL)
logging.getLogger("requests.packages.urllib3.connectionpool").setLevel(logging.CRITICAL)
logging.getLogger("requests.packages.urllib3").setLevel(logging.CRITICAL)
logging.getLogger("matplotlib").setLevel(logging.CRITICAL)
logging.getLogger("werkzeug").setLevel(logging.CRITICAL)
logging.getLogger('transformers').setLevel(logging.CRITICAL)
logging.getLogger('numexpr').setLevel(logging.CRITICAL)
logging.getLogger('urllib3.connectionpool').setLevel(logging.CRITICAL)
logging.Logger.manager.loggerDict['requests'].setLevel(logging.CRITICAL)

LOG_LEVEL = getattr(logging, os.environ["LOG_LEVEL"]) if "LOG_LEVEL" in os.environ else logging.DEBUG
logging.basicConfig(level=LOG_LEVEL, stream=sys.stdout, filemode='w',
                    format='"%(process)d >>> %(asctime)s - %(levelname)s - %(funcName)s: %(message)s"')
logger = logging.getLogger("src.utils.logger")
