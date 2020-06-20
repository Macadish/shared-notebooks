import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.debug('This message should go to the log file')
logging.info('So should this')
logging.warning('And this, too')
try:
    assert(1==2)
except Exception as e:
    logging.error("Exception occurred", exc_info=True)
    logging.exception()
    raise
