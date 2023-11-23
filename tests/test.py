import logging

logging.basicConfig(
    level=logging.ERROR,
    format='%(levelname)s - %(message)s - \n \
        Location : %(pathname)s:%(lineno)d:%(funcName)s',
    # filename='example.log'
)



logging.error("This is a debug message.")