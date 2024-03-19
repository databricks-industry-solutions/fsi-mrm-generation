import logging

import requests

from databricks.mrm_objects import *
from databricks.mrm_utils import *

logger = logging.getLogger('databricks')

a_bit_of_space = '<div class="col-xs-12" style="height:100px;"></div>'
a_little_bit_of_space = '<div class="col-xs-12" style="height:70px;"></div>'


class ModelRiskApi:

    def __init__(self, base_url, api_token):

        self.base_url = base_url
        self.headers = {'Authorization': f'Bearer {api_token}'}

    def __submit_api_call(self, url):
        try:
            response = requests.get(url=url, headers=self.headers)
            response.raise_for_status()  # this will raise an error if the request failed
        except requests.exceptions.RequestException as e:
            logger.error(f"Request to {url} failed: {e}")
            raise

        try:
            data = response.json()  # this will raise an error if the response isn't valid JSON
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON from response: {e}")
            raise

        if 'error_code' in data:
            raise Exception(f"Error while retrieving content, [{data}]")
        return data

    @staticmethod
    def __extract_run_libraries(run_tags):
        """
        Extracting the list of libraries captured with mlflow experiment.
        Note that MLFlow will struggle to capture notebook scoped libraries (using %pip install).
        We highly recommend the use of cluster scope libraries for heightened governance
        :param run_tags: The list of key /value tags captured by MLFlow
        :return: The list of libraries
        """
        libraries_objects = []
        if 'mlflow.databricks.cluster.libraries' in run_tags:
            libraries = json.loads(run_tags['mlflow.databricks.cluster.libraries'])
            for installable in libraries['installable']:
                library_type = list(installable.keys())[0]
                library_coord = installable[library_type]
                if library_type == 'maven':
                    library_coord = library_coord['coordinates']
                    libraries_objects.append(Library('maven', library_coord))
            for redacted in libraries['redacted']:
                library_type = list(redacted.keys())[0]
                library_coord = redacted[library_type]
                if library_type == 'maven':
                    library_coord = library_coord['coordinates']
                    libraries_objects.append(Library('maven', library_coord))
        return Libraries(libraries_objects)

    @staticmethod
    def __extract_run_cluster(run_tags):
        """
        Extracting cluster information as captured by MLFlow. This tag (stored as JSON string) seems to capture
        all information required without the need to pull additional information from cluster API.
        :param run_tags: The list of key /value tags captured by MLFlow
        :return: cluster information such as name, DBR and instance type
        """
        cluster_info = run_tags.get('mlflow.databricks.cluster.info') or None
        if cluster_info:
            cluster_info = json.loads(cluster_info)
            autoscale = cluster_info.get('autoscale') or None
            if autoscale:
                num_workers = f'{autoscale.get("min_workers")}-{autoscale.get("max_workers")}'
            else:
                num_workers = cluster_info.get('autoscale') or None
            return ExperimentCluster(
                cluster_info.get('cluster_name') or None,
                cluster_info.get('spark_version') or None,
                cluster_info.get('node_type_id') or None,
                num_workers
            )
        else:
            return None

    @staticmethod
    def __extract_run_data_sources(run_object, run_tags):
        """
        Simply extract all data sources captured by MLFlow. Data sources are captured as comma separated entries.
        :param run_object: The list of key /value tags captured by MLFlow
        :return: The list of data sources captured by MLFlow
        """
        mlflow_data_records = []
        if 'inputs' in run_object:
            inputs = run_object['inputs']
            if 'dataset_inputs' in inputs and len(inputs['dataset_inputs']) > 0:
                for dataset_input in inputs['dataset_inputs']:
                    dataset = dataset_input['dataset']
                    if dataset['source_type'] == 'delta_table':
                        delta_record = json.loads(dataset['source'])
                        mlflow_data_records.append(ExperimentDataSource(
                            delta_record['delta_table_name'],
                            'delta',
                            dataset['name'].split('@')[-1],
                        ))
                    else:
                        mlflow_data_records.append(ExperimentDataSource(
                            dataset['name'].split('@')[0],
                            dataset['source_type'],
                            dataset['name'].split('@')[-1],
                        ))
        else:
            mlflow_data = run_tags.get('sparkDatasourceInfo') or None
            if mlflow_data:
                for source_record in mlflow_data.split('\n'):
                    source_record_dict = {}
                    source_record_kv_pairs = source_record.split(',')
                    for source_record_kv in source_record_kv_pairs:
                        kv = source_record_kv.split('=')
                        source_record_dict[kv[0]] = kv[1]

                    name = source_record_dict.get('path') or None
                    fmt = source_record_dict.get('format') or None
                    version = source_record_dict.get('version') or None

                    mlflow_data_records.append(ExperimentDataSource(
                        name,
                        fmt,
                        version,
                    ))

        if len(mlflow_data_records) > 0:
            return ExperimentDataSources(mlflow_data_records)
        else:
            return None

    @staticmethod
    def __extract_run_artifact_signature(fields):
        """
        Return model input and output signature, as captured by MLFlow or manually registered by end user.
        We extract the field name and type for each input or output feature we can later represent as a graph
        :param fields: The input or output fields, stored as JSON string in MLFlow tag.
        :return: The parsed field in the form of [name:type]
        """
        fields = json.loads(fields)
        parameters = []
        for field in fields:
            if field['type'] == 'tensor':
                tensor_type = field['tensor-spec']['dtype']
                field_type = f'tensor[{tensor_type}]'
            else:
                field_type = field["type"]
            if 'name' in field:
                parameters.append(f'{field["name"]} [{field_type}]')
            else:
                parameters.append(f'[{field_type}]')
        return parameters

    @staticmethod
    def __extract_run_artifact_flavors(flavors):
        """
        Whether those are native python models or ML frameworks (keras, sklearn, xgboost), models may have been
        serialized (pickled) using different flavors. We retrieve all artifacts logged together with their
        interpreter version in order to guarantee model reproducibility.
        :param flavors: The logged artifacts as MLFlow tags
        :return: the list of logged artifacts, flavors and interpreter versions.
        """
        logged_flavors = []
        for flavor in flavors:
            executor_type = flavor
            version = list(filter(lambda x: x.endswith('version'), flavors[flavor].keys()))
            if version:
                executor_version = flavors[flavor][version[0]]
            else:
                executor_version = None
            logged_flavors.append(ArtifactFlavor(executor_type, executor_version))
        return logged_flavors

    def __extract_run_artifacts(self, run_tags):
        """
        MLFlow captured all artifacts for the given model experiment. Artifacts may include model input and output
        signatures as well as interpreter versions and ML frameworks.
        :param run_tags: The tag captured on MLFlow as JSON string.
        :return: the list of artifacts together with model signature
        """
        model_info = run_tags['mlflow.log-model.history']
        fmt = '%Y-%m-%d %H:%M:%S.%f'

        if model_info:
            artifacts = []
            model_info = json.loads(model_info)
            for model_logged in model_info:
                created = datetime.datetime.strptime(model_logged['utc_time_created'], fmt)
                if 'flavors' in model_logged:
                    flavors = self.__extract_run_artifact_flavors(model_logged['flavors'])
                else:
                    flavors = []
                if 'signature' in model_logged:
                    signature = model_logged['signature']
                    signature_input = self.__extract_run_artifact_signature(signature['inputs'])
                    signature_output = self.__extract_run_artifact_signature(signature['outputs'])
                    signature = ArtifactSignature(signature_input, signature_output)
                else:
                    signature = None
                artifacts.append(Artifact(created, flavors, signature))
            return Artifacts(artifacts)
        else:
            return None

    @staticmethod
    def __extract_source_lineage_name(data_source):
        """
        Data lineage may return different upstream data sources. Only supporting tables for now, those sources
        will include information of catalog, database and table. We could possibly extend this function to return
        actual schema and column lineage, but let's keep it simple for now.
        :param data_source: the source captured in data lineage from UC
        :return: the parsed datasource returned in a 3 layer namespace form (catalog.database.table)
        """
        if 'tableInfo' in data_source:
            table_info = data_source['tableInfo']
            coordinate = []
            if 'catalog_name' in table_info:
                coordinate.append(table_info['catalog_name'])
            if 'schema_name' in table_info:
                coordinate.append(table_info['schema_name'])
            else:
                coordinate.append('default')
            if 'name' in table_info:
                coordinate.append(table_info['name'])
                return 'table', '.'.join(coordinate)
            else:
                logger.warning("No table name found, ignoring upstream")
        elif 'fileInfo' in data_source:
            file_info = data_source['fileInfo']
            file_path = file_info['path']
            return 'path', file_path
        else:
            logger.warning("Unsupported format for source {}".format(data_source.keys()))
        return None, None

    @staticmethod
    def __extract_notebook(html_content):
        """
        When exported as HTML, notebook contain multiple metadata, complex HTML, and actual notebook content stored
        as base 64 encoded string. This function will retrieve only notebook content from HTML notebook.
        :param html_content: the raw HTML content for exported notebook
        :return: the actual notebook content as base 64 encoded, wrapped into a class for HTML display
        """
        notebook_regex = re.compile(
            "DATABRICKS_NOTEBOOK_MODEL = '((?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?)'"
        )
        matches = notebook_regex.findall(html_content)
        if matches:
            return Notebook(matches[0])
        else:
            logger.error("Could not extract notebook content from HTML")
            return None

    def __process_lineage(self, response):
        """
        Extracting data lineage is a recursive process. For each data source, we need to extract information of
        the source itself (the name, typically in 3 layer namespace) as well as its upstream sources. This will go
        through the UC API again till no upstream source can be found. As UC will grow, we need to cover additional
        requirement such as process lineage, dashboard lineage, etc.
        :param response: the original response from UC API
        :return: the result of the recursion for 1 given data source as a list of upstream dependencies.
        """
        children = []
        if 'upstreams' in response:
            upstreams = response['upstreams']
            for upstream in upstreams:
                upstream_type, upstream_source = self.__extract_source_lineage_name(upstream)
                upstream_lineage = self.__get_lineage_rec(upstream_source, upstream_type)
                children.append(upstream_lineage)
        return children

    @staticmethod
    def __process_model_parent(response):
        """
        Core business logic to extract information from the model registered on MLFlow. This higher level taxonomy
        should capture information around model owner, timestamp as well as all latest versions (for each stage).
        :param response: the original JSON response from MLFlow API
        :return: the metadata of model as registered on MLFlow together with raw information for each version
        """
        response = response['registered_model']
        model_parent_owner = response['user_id']
        model_catalog = response['name'].split('.')[0]
        model_schema = response['name'].split('.')[1]
        model_name = response['name'].split('.')[2]
        model_parent_creation = parse_date(response['creation_timestamp'])
        if 'description' not in response:
            model_parent_description = None
        else:
            model_parent_description = ModelDescription(response['description'])
        model_parent_tags = extract_tags(response)
        model_parent_aliases = extract_aliases(response)

        return ModelParent(
            model_catalog,
            model_schema,
            model_name,
            model_parent_description,
            model_parent_tags,
            model_parent_owner,
            model_parent_creation,
            model_parent_aliases
        )

    def __process_model_run(self, response, run_id):
        """
        Core business logic to extract information for a given model experiment. This 3rd layer taxonomy will contain
        vital information about the technical context behind this model submission such as the cluster dependencies,
        libraries and associated code. Depending on the type of processing (JOB or INTERACTIVE), we will pull the
        relevant information and tags.
        :param response: the original JSON response from MLFlow experiment tracker API
        :param run_id: the experiment ID captured by MLFlow registry.
        :return: the technical context surrounding this model submission, wrapped as class for HTML output.
        """

        run_object = response['run']
        run_data = run_object['data']
        run_info = run_object['info']
        run_tags = key_value_to_dict(run_data['tags'])

        run_experiment_id = run_info['experiment_id']
        run_timestamp = parse_time(run_info['start_time'])
        run_parent_id = run_tags.get('mlflow.parentRunId') or None
        run_description = run_tags.get('mlflow.note.content') or None
        run_user = run_tags.get('mlflow.user') or None
        run_workspace = run_tags.get('mlflow.databricks.workspaceURL') or None

        run_data_sources = self.__extract_run_data_sources(run_object, run_tags)
        run_cluster = self.__extract_run_cluster(run_tags)
        run_artifacts = self.__extract_run_artifacts(run_tags)
        run_libraries = self.__extract_run_libraries(run_tags)

        if run_description:
            run_description = ExperimentDescription(run_description)

        if 'mlflow.source.type' in run_tags.keys():
            source_type = run_tags.get('mlflow.source.type')
            if source_type == 'NOTEBOOK':
                # Pull associated notebook
                # TODO: use revision ID
                source_name = run_tags.get('mlflow.source.name')
                source_code = self.__get_notebook(source_name)
                source_commit = run_tags.get('mlflow.databricks.notebookRevisionID')
            elif source_type == 'JOB':
                # Pull associated JOB output
                source_name = run_tags.get('mlflow.source.name')
                source_code = self.__get_notebook_from_job(source_name)
                source_commit = None
            else:
                source_name = None
                source_code = None
                source_commit = None

            if 'mlflow.databricks.gitRepoUrl' in run_tags:
                source_url = run_tags.get('mlflow.databricks.gitRepoUrl')
                source_commit = run_tags.get('mlflow.databricks.gitRepoCommit')
            else:
                source_url = None
        else:
            source_type = None
            source_name = None
            source_commit = None
            source_url = None
            source_code = None

        if 'params' in run_data:
            run_params = ExperimentParameters(key_value_to_dict(run_data['params']))
        else:
            run_params = {}

        if 'metrics' in run_data:
            run_metrics = ExperimentMetrics(key_value_to_dict(run_data['metrics']))
            # TODO: consider extracting metrics across runs or find way to vizualize in parallel graph
        else:
            run_metrics = {}

        return ModelExperiment(
            run_id,
            run_description,
            run_user,
            run_workspace,
            run_experiment_id,
            run_timestamp,
            run_params,
            run_metrics,
            run_data_sources,
            run_parent_id,
            source_type,
            source_name,
            source_url,
            source_commit,
            source_code,
            run_cluster,
            run_artifacts,
            run_libraries
        )

    @staticmethod
    def __process_model_versions(data, model_name, model_parent_aliases):
        versions = []
        for model_version in data['model_versions']:
            model_date = parse_date(model_version['created_at'])
            user = model_version['created_by']
            versions.append(ModelPreviousVersion(
                model_version['version'],
                model_date,
                user,
                model_parent_aliases.get(model_version['version'])
            ))
        return ModelVersions(model_name, versions)

    @staticmethod
    def __process_model_version(data, model_version):
        """
        Core business logic to extract information from a given model version. This secondary level taxonomy should
        capture information about model submitter, the submission data as well as the desired transition state (e.g.
        from STAGING to PROD).
        :param data: the original JSON response from MLFlow API
        :param model_version: the version of the model to fetch from MLFlow transition API.
        :return: the metadata of the model version as submitted by end user, wrapped as class for HTML output.
        """
        data = data['model_version']
        if 'description' not in data:
            description = None
        else:
            description = ModelDescription(data['description'])
        return ModelSubmission(
            description,
            data['user_id'],
            model_version,
            parse_time(data['creation_timestamp']),
            data['run_id'],
            extract_tags(data)
        )

    def __get_model_version(self, model_name, model_version):
        """
        Entry point for a given model risk management submission. We retrieve information surrounding a particular
        version of a model as registered on MLFlow. Note that this version should be the latest available for a given
        stage (e.g. STAGING).
        :param model_name: the name of the model to fetch from MLFlow API.
        :param model_version: the version of the model to fetch from MLFlow API response (optional).
        :return:
        """
        logger.info(f'Retrieving version [{model_version}] for model [{model_name}]')
        url = f'{self.base_url}/api/2.0/mlflow/unity-catalog/model-versions/get?name={model_name}&version={model_version}'
        data = self.__submit_api_call(url)
        return self.__process_model_version(data, model_version)

    def __get_model_versions(self, model_name, model_parent_aliases):
        logger.info(f'Retrieving versions for model [{model_name}]')
        url = f'{self.base_url}/api/2.1/unity-catalog/models/{model_name}/versions'
        data = self.__submit_api_call(url)
        return self.__process_model_versions(data, model_name, model_parent_aliases)

    def __get_model_parent(self, model_name):
        """
        Entry point for a given Model risk management output. We pulled information from MLFlow registry server for a
        given model name, regardless of its desired version. Information returned will include business context around
        our model such as creation timestamp, model owner, etc.
        :param model_name: the name of the model to fetch from MLFlow API.
        :return: The business context surrounding our model, as captured by MLFlow directly or filled by end user.
        """
        logger.info(f'Retrieving model [{model_name}]')
        url = f'{self.base_url}/api/2.0/mlflow/unity-catalog/registered-models/get?name={model_name}'
        data = self.__submit_api_call(url)
        return self.__process_model_parent(data)

    def __get_model_run(self, run_id):
        """
        Entry point for a given model experiment. Querying the MLFlow tracker API, we aim at extracting all technical
        context surrounding a given model registered on MLFlow, including notebook, data sources, cluster dependencies
        :param run_id: the ID of the experiment to fetch from MLFlow
        :return: the technical context surrounding a given model, returned as wrapped class for HTML output
        """
        logger.info(f'Retrieving run_id [{run_id}] associated to model')
        url = f'{self.base_url}/api/2.0/mlflow/runs/get?run_id={run_id}'
        data = self.__submit_api_call(url)
        return self.__process_model_run(data, run_id)

    def __get_lineage_rec(self, upstream_source, upstream_type):
        """
        Querying the UC API, we retrieve all data lineage for each data source captured on MLFlow experiment. This
        requires call that same API recursively to fetch all upstream dependencies.
        :param upstream_source: the name of the data source to fetch from table API, in the form of 3 layer namespace
        :param upstream_type: the type of the data source to fetch from table API, in the form of 3 layer namespace
        :return: the entire upstream lineage wrapped as a class for HTML output
        """
        children = []
        url = f'{self.base_url}/api/2.0/lineage-tracking/table-lineage?table_name={upstream_source}'
        if upstream_type == 'delta':
            response = json.loads(requests.get(url=url, headers=self.headers).text)
            if 'error_code' in response:
                logger.error(f"Error in lineage response, {response['message']}")
            else:
                children = self.__process_lineage(response)
        return LineageDataSource(upstream_source, upstream_type, children)

    def __get_lineage(self, data_source_names, model_name):
        """
        Entry point for data lineage, we wrapped all data sources and their lineage into a class object to facilitate
        HTML creation at later stage
        :param data_source_names: the list of all source of data captured by MLFlow and available as such on UC
        :return: the entire upstream lineage wrapped as a class for HTML output
        """
        logger.info(f'Retrieving data lineage for {len(data_source_names)} data source(s)')
        return Lineage([self.__get_lineage_rec(data_source_name, data_source_type) for data_source_name, data_source_type in data_source_names], model_name)

    def __get_notebook_from_job(self, job_id):
        """
        Querying the JOB API, we aim at extracting notebook content associated with a MLFlow experiment.
        :param job_id: the ID of job to fetch notebook output from
        :return: the notebook content returned as encoded base 64
        """
        run_id = job_id.split('/')[-1]
        url = f'{self.base_url}/api/2.1/jobs/runs/export?run_id={run_id}&views_to_export=CODE'
        response = json.loads(requests.get(url=url, headers=self.headers).text)
        if 'error_code' in response:
            logger.error(f"Error in job response, {response['message']}")
            return None
        content = list(filter(lambda x: x['type'] == 'NOTEBOOK', response['views']))
        if len(content) > 0:
            html_org_content = content[0]['content']
            return self.__extract_notebook(html_org_content)
        else:
            logger.error(f"Could not find any output content for job [{job_id}]")
            return None

    def __get_notebook(self, remote_path):
        """
        Querying the workspace API, we aim at extracting notebook content associated with a MLFlow experiment.
        :param remote_path: the path of the notebook to fetch
        :return: the notebook content returned as encoded base 64
        """
        logger.info(f'Retrieving notebook [{remote_path}] associated to model run')
        url = f'{self.base_url}/api/2.0/workspace/export?path={remote_path}&format=HTML&direct_download=False'
        response = json.loads(requests.get(url=url, headers=self.headers).text)
        if 'error_code' in response:
            logger.error(f"Error in notebook response, {response['message']}")
            return None
        html_org_content = str(base64.b64decode(response['content']), 'utf-8')
        return self.__extract_notebook(html_org_content)

    def generate_doc(self, model_name, output_file, model_version, verbatim_file=None):
        """
        Public entry point for model risk management PDF output. Given a model name, an optional model version and a
        target output file, we will fetch all the required information from various databricks API, bring that
        technical and business context together and generate PDF output accordingly. After multiple consideration
        being the use of e.g. LateX library, we decided to leverage HTML as main format as it supports markdown
        information, HTML that we can "beautify" using boostrap CSS and convert to PDF document.
        :param model_name: the name of the model to fetch from databricks
        :param output_file:  the output file to write PDF document 
        :param model_version: the version of the model to fetch from databricks
        :param verbatim_file: giving user the opportunity to supply their own verbatim files instead of default
        :return:
        """

        if model_version:
            logger.info(f'Generating MRM output for model [{model_name}] (v{model_version})')
        else:
            logger.info(f'Generating MRM output for latest model [{model_name}]')

        # load our text
        verbatim = load_verbatim(verbatim_file)

        # retrieve model from Unity Catalog
        model_parent = self.__get_model_parent(model_name)
        model_versions = self.__get_model_versions(model_name, model_parent.model_parent_aliases)
        model_version = self.__get_model_version(model_name, model_version)
        model_run = self.__get_model_run(model_version.model_run_id)

        # retrieve model data input and lineage
        if model_run.run_data_sources:
            data_sources = model_run.run_data_sources
            data_lineage = self.__get_lineage(data_sources.sources(), model_name)
        else:
            data_sources = None
            data_lineage = None

        ##########################################################################################
        # FRONT PAGE OF OUR REPORT
        ##########################################################################################

        html = [
            f'<h1 class="text-center">{verbatim["title"]}</h1>',
            '<h3 class="card-subtitle text-muted text-center">{}</h3>'.format(model_name.split('.')[-1]),
        ]

        ##########################################################################################
        # TOP LEVEL SECTION
        # Include information about model ownership and description
        ##########################################################################################

        html.extend([
            '<div class="section section-break section-content">',
            f'<h1>{verbatim["mlflow_model"]["header"]}</h1>',
            f'<p>{verbatim["mlflow_model"]["info"]}</p>',
            a_bit_of_space
        ])

        html.extend(model_parent.to_html(h_level=1))
        html.append(a_little_bit_of_space)

        if model_parent.model_description:
            html.append('<small class="text-muted">description from mlflow registry</small>')
            html.extend(model_parent.model_description.to_html(h_level=2))
        else:
            html.extend([
                '<div class="alert alert-warning section-content" role="alert">',
                f'<p>{verbatim["mlflow_model"]["error"]}</p>',
                '</div>'
            ])

        ##########################################################################################
        # MODEL HISTORY SECTION
        # Include model submission request, triggering independent validation
        ##########################################################################################

        html.extend([
            '<div class="section section-break section-content">',
            f'<h1>{verbatim["mlflow_model_versions"]["header"]}</h1>',
            f'<p>{verbatim["mlflow_model_versions"]["info"]}</p>',
            a_bit_of_space
        ])

        html.extend(model_versions.to_html(h_level=1))

        ##########################################################################################
        # MODEL SUBMISSION SECTION
        # Include model submission request, triggering independent validation
        ##########################################################################################

        html.extend([
            '<div class="section section-break section-content">',
            f'<h1>{verbatim["mlflow_model_version"]["header"]}</h1>',
            f'<p>{verbatim["mlflow_model_version"]["info"]}</p>',
            a_bit_of_space
        ])

        html.extend(model_version.to_html(h_level=1))
        html.append(a_little_bit_of_space)
        if model_version.model_description:
            html.append('<small class="text-muted">description from mlflow registry</small>')
            html.extend(model_version.model_description.to_html(h_level=2))
        else:
            html.extend([
                '<div class="alert alert-warning section-content" role="alert">',
                f'<p>{verbatim["mlflow_model_version"]["error"]}</p>',
                '</div>'
            ])

        ##########################################################################################
        # MODEL EXPERIMENT SECTION
        # This section will get into the weeds of the experiment itself, technical context
        ##########################################################################################

        html.extend([
            '<div class="section section-break section-content">',
            f'<h1>{verbatim["mlflow_model_version_run"]["header"]}</h1>',
            f'<p>{verbatim["mlflow_model_version_run"]["info"]}</p>',
            a_bit_of_space
        ])

        html.extend(model_run.to_html(h_level=1))
        html.append(a_little_bit_of_space)
        if model_run.run_description:
            html.append('<small class="text-muted">description from mlflow experiment</small>')
            html.extend(model_run.run_description.to_html(h_level=2))
        else:
            html.extend([
                '<div class="alert alert-warning section-content" role="alert">',
                f'<p>{verbatim["mlflow_model_version_run"]["error"]}</p>',
                '</div>'
            ])

        ##########################################################################################
        # MODEL IMPLEMENTATION HEADER
        ##########################################################################################

        html.extend([
            '<div class="section section-break section-content">',
            f'<h1>{verbatim["implementation"]["header"]}</h1>',
            f'<p>{verbatim["implementation"]["info"]}</p>',
            '</div>'
        ])

        ##########################################################################################
        # MODEL ARTIFACTS
        # We list all artifacts logged alongside our model
        ##########################################################################################

        html.extend([
            '<div class="section section-content">',
            f'<h2>{verbatim["implementation_artifacts"]["header"]}</h2>',
            f'<p>{verbatim["implementation_artifacts"]["info"]}</p>',
            '</div>'
        ])
        if model_run.run_artifacts:
            html.extend(model_run.run_artifacts.to_html(h_level=3))
        else:
            html.extend([
                '<div class="alert alert-warning section-content" role="alert">',
                f'<p>{verbatim["implementation_artifacts"]["error"]}</p>'
                '</div>'
            ])

        ##########################################################################################
        # MODEL NOTEBOOKS
        # We report the output of the job / notebook to bring all necessary context
        ##########################################################################################

        html.extend([
            '<div class="section section-break section-content">',
            f'<h2>{verbatim["implementation_approach"]["header"]}</h2>',
            f'<p>{verbatim["implementation_approach"]["info"]}</p>',
            '</div>'
        ])
        if model_run.source_code:
            html.extend(model_run.source_code.to_html(h_level=2))
        else:
            html.extend([
                '<div class="alert alert-warning section-content" role="alert">',
                f'<p>{verbatim["implementation_approach"]["error"]}</p>'
                '</div>'
            ])

        ##########################################################################################
        # MODEL PARAMETERS
        # We report all parameters logged on mlflow, either programmatically (autologging) or not
        ##########################################################################################

        html.extend([
            '<div class="section section-break section-content">',
            f'<h2>{verbatim["model_parameters"]["header"]}</h2>',
            f'<p>{verbatim["model_parameters"]["info"]}</p>',
            '</div>'
        ])
        if model_run.run_params:
            html.extend(model_run.run_params.to_html(h_level=3))
        else:
            html.extend([
                '<div class="alert alert-warning section-content" role="alert">',
                f'<p>{verbatim["model_parameters"]["error"]}</p>'
                '</div>'
            ])

        ##########################################################################################
        # MODEL METRICS
        # We report all metrics logged on mlflow, either programmatically (autologging) or not
        ##########################################################################################

        html.extend([
            '<div class="section section-break section-content">',
            f'<h2>{verbatim["model_metrics"]["header"]}</h2>',
            f'<p>{verbatim["model_metrics"]["info"]}</p>',
            '</div>'
        ])
        if model_run.run_metrics:
            html.extend(model_run.run_metrics.to_html(h_level=3))
        else:
            html.extend([
                '<div class="alert alert-warning section-content" role="alert">',
                f'<p>{verbatim["model_metrics"]["error"]}</p>'
                '</div>'
            ])

        ##########################################################################################
        # MODEL DEPENDENCIES HEADER
        ##########################################################################################

        html.extend([
            '<div class="section section-break section-content">',
            f'<h1>{verbatim["model_dependencies"]["header"]}</h1>',
            f'<p>{verbatim["model_dependencies"]["info"]}</p>',
            '</div>'
        ])

        ##########################################################################################
        # MODEL INFRASTRUCTURE SECTION
        # We report infrastructure dependency of our model
        ##########################################################################################

        html.extend([
            '<div class="section section-content">',
            f'<h2>{verbatim["model_dependencies_infra"]["header"]}</h2>',
            f'<p>{verbatim["model_dependencies_infra"]["info"]}</p>',
            '</div>'
        ])
        if model_run.run_cluster:
            html.extend(model_run.run_cluster.to_html(h_level=2))
        else:
            html.extend([
                '<div class="alert alert-warning section-content" role="alert">',
                f'<p>{verbatim["model_dependencies_infra"]["info"]}</p>'
                '</div>'
            ])

        ##########################################################################################
        # MODEL LIBRARY SECTION
        # We report all libraries logged alongside our model
        ##########################################################################################

        html.extend([
            '<div class="section section-content">',
            f'<h2>{verbatim["model_dependencies_libraries"]["header"]}</h2>',
            f'<p>{verbatim["model_dependencies_libraries"]["info"]}</p>',
            '</div>'
        ])

        if model_run.run_libraries and model_run.run_libraries.non_empty():
            html.extend(model_run.run_libraries.to_html(h_level=2))
        else:
            html.extend([
                '<div class="alert alert-warning section-content" role="alert">',
                f'<p>{verbatim["model_dependencies_libraries"]["error"]}</p>'
                '</div>'
            ])

        ##########################################################################################
        # MODEL SIGNATURE SECTION
        # We report all model input and output signature
        ##########################################################################################

        html.extend([
            '<div class="section section-break section-content">',
            f'<h2>{verbatim["model_signature"]["header"]}</h2>',
            f'<p>{verbatim["model_signature"]["info"]}</p>',
            '</div>'
        ])
        if model_run.run_artifacts:
            for artifact in model_run.run_artifacts.artifacts:
                if artifact.signature:
                    html.extend(artifact.signature.to_html(h_level=3))
        else:
            html.extend([
                '<div class="alert alert-warning section-content" role="alert">',
                f'<p>{verbatim["model_signature"]["error"]}</p>'
                '</div>'
            ])

        ##########################################################################################
        # MODEL DATA DEPENDENCY SECTION
        # We report all data sources logged on MLFlow
        ##########################################################################################

        html.extend([
            '<div class="section section-content">',
            f'<h2>{verbatim["model_dependencies_data"]["header"]}</h2>',
            f'<p>{verbatim["model_dependencies_data"]["info"]}</p>',
            '</div>'
        ])

        if data_sources:
            html.extend(data_sources.to_html(h_level=2))
        else:
            html.extend([
                '<div class="alert alert-warning section-content" role="alert">',
                f'<p>{verbatim["model_dependencies_data"]["error"]}</p>'
                '</div>'
            ])

        ##########################################################################################
        # MODEL DATA LINEAGE SECTION
        # We report a graph representation of our data lineage
        ##########################################################################################

        html.extend([
            '<div class="section section-break section-content">',
            f'<h2>{verbatim["model_dependencies_lineage"]["header"]}</h2>',
            f'<p>{verbatim["model_dependencies_lineage"]["info"]}</p>',
            '</div>'
        ])

        if data_lineage:
            html.extend(data_lineage.to_html())
        else:
            html.extend([
                '<div class="alert alert-warning section-content" role="alert">',
                f'<p>{verbatim["model_dependencies_lineage"]["error"]}</p>'
                '</div>'
            ])

        ##########################################################################################
        # GENERATE PDF OUTPUT
        # We pimp our HTML with additional CSS and HTML template and generate PDF accordingly
        ##########################################################################################

        generate_pdf(html, output_file)

        ##########################################################################################
        # UPLOAD MODEL PDF TO VERSION
        ##########################################################################################
