import argparse
import logging
import sys

from databricks.sdk import WorkspaceClient

from databricks.mrm import ModelRiskApi


def _configure_logging():
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)
    return logging.getLogger('databricks')


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog='mrm-generate',
        description='Generate a model risk management PDF from a Unity Catalog model.',
    )
    parser.add_argument("--db-workspace", help="databricks workspace url (optional, auto-resolved from environment)")
    parser.add_argument("--db-token", help="databricks token (optional, auto-resolved from environment)")
    parser.add_argument("--db-profile", help="databricks config profile name (optional)")
    parser.add_argument("--model-name", help="unity catalog model name (catalog.schema.model)")
    parser.add_argument("--model-version", help="model version")
    parser.add_argument("--output", help="pdf output file")
    return parser.parse_args(argv)


def main(argv=None):
    logger = _configure_logging()
    args = _parse_args(argv)

    #######################
    # READING CONFIGURATION
    #######################

    if not args.model_name:
        logger.error("please provide a model name (fully qualified: catalog.schema.model)")
        sys.exit(1)

    if not args.output:
        logger.error("please provide a output file location for pdf document")
        sys.exit(1)

    if args.model_version:
        model_version = int(args.model_version)
    else:
        model_version = None

    ###################################
    # INITIALIZE DATABRICKS SDK CLIENT
    ###################################

    # WorkspaceClient auto-resolves authentication from environment variables,
    # config profiles (~/.databrickscfg), Azure/GCP managed identity, or
    # explicit host/token. This replaces the legacy PAT-only approach.

    client_kwargs = {}
    if args.db_workspace:
        client_kwargs['host'] = args.db_workspace
    if args.db_token:
        client_kwargs['token'] = args.db_token
    if args.db_profile:
        client_kwargs['profile'] = args.db_profile

    try:
        client = WorkspaceClient(**client_kwargs)
        # Validate connectivity
        client.current_user.me()
    except Exception as e:
        logger.error(f"Failed to authenticate with Databricks: {e}")
        logger.error("Ensure credentials are configured via environment variables "
                     "(DATABRICKS_HOST, DATABRICKS_TOKEN), config profile, or CLI arguments.")
        sys.exit(1)

    #####################
    # GENERATING MRM FILE
    #####################

    api = ModelRiskApi(client)
    api.generate_doc(args.model_name, args.output, model_version)


if __name__ == '__main__':
    main()
