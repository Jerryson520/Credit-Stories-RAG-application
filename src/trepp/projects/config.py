"""
config
"""

import os


# environment specific
DSBUCKETNAME = os.environ.get('DSBUCKETNAME')
INPUTBUCKETNAME = os.environ.get('INPUTBUCKETNAME')
OUTPUTBUCKETNAME = os.environ.get('OUTPUTBUCKETNAME')
DEPLOYENV = os.environ.get('DEPLOYENV')

# trained model information
__version__ = '0.0.1'
