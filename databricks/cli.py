import argparse
import logging
import os
import sys

from databricks.mrm import ModelRiskApi
from databricks.sdk import WorkspaceClient


def _ensure_native_lib_path():
    """Make WeasyPrint's native libraries loadable on macOS.

    WeasyPrint loads Pango/GLib/GDK-PixBuf via ``dlopen``. On macOS those live
    under Homebrew's prefix, which the dynamic loader does not search by default,
    and the ``DYLD_*`` variables that would add it are only read at process
    launch. If Homebrew's lib dir is missing from the search path, add it and
    re-exec this process once so the libraries resolve without the user having to
    export anything. No-op on non-macOS platforms and when already configured.
    """
    if sys.platform != 'darwin':
        return
    lib_dirs = [d for d in ('/opt/homebrew/lib', '/usr/local/lib') if os.path.isdir(d)]
    if not lib_dirs:
        return
    search = os.environ.get('DYLD_FALLBACK_LIBRARY_PATH', '')
    parts = search.split(os.pathsep) if search else []
    if all(d in parts for d in lib_dirs):
        return  # already on the search path — avoids a re-exec loop
    os.environ['DYLD_FALLBACK_LIBRARY_PATH'] = os.pathsep.join(lib_dirs + parts)
    os.execv(sys.executable, [sys.executable, *sys.argv])


def _configure_logging():
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)

    # WeasyPrint emits a large volume of INFO progress lines and benign CSS
    # warnings (unsupported Bootstrap vendor rules, media queries, etc.). Keep
    # only genuine errors so they don't drown out our own output.
    logging.getLogger('weasyprint').setLevel(logging.ERROR)
    logging.getLogger('fontTools').setLevel(logging.ERROR)

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
    _ensure_native_lib_path()
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
