import argparse
import logging
import sys

from databricks.mrm import ModelRiskApi


root = logging.getLogger()
root.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)
logger = logging.getLogger('databricks')

parser = argparse.ArgumentParser()
parser.add_argument("--db-workspace", help="databricks workspace url")
parser.add_argument("--db-token", help="databricks token")
parser.add_argument("--model-name", help="mlflow model name")
parser.add_argument("--model-version", help="mlflow model version")
parser.add_argument("--output", help="pdf output file")

args = parser.parse_args()

#######################
# READING CONFIGURATION
#######################

if not args.db_workspace:
    logger.error("please provide databricks workspace url")
    sys.exit(1)

if not args.db_token:
    logger.error("please provide databricks token")
    sys.exit(1)

if not args.model_name:
    logger.error("please provide a model name")
    sys.exit(1)

if not args.output:
    logger.error("please provide a output file location for pdf document")
    sys.exit(1)

if args.model_version:
    model_version = int(args.model_version)
else:
    model_version = None

#####################
# GENERATING MRM FILE
#####################

api = ModelRiskApi(args.db_workspace, args.db_token)
api.generate_doc(args.model_name, args.output, model_version)
