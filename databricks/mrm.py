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
        self.api_run = f'{base_url}/api/2.0/mlflow/runs/get'
        self.api_experiment = f"{base_url}/api/2.0/mlflow/experiments/get"
        self.api_search = f"{base_url}/api/2.0/mlflow/registered-models/search"
        self.api_lineage = f"{base_url}/api/2.0/lineage-tracking/table-lineage"
        self.api_artifact = f"{base_url}/api/2.0/mlflow/artifacts/list"
        self.api_job_list = f"{base_url}/api/2.1/jobs/runs/list"
        self.api_job_export = f"{base_url}/api/2.1/jobs/runs/export"
        self.api_registry = f'{base_url}/api/2.0/mlflow/databricks/registered-models/get'
        self.api_transitions = f'{base_url}/api/2.0/mlflow/transition-requests/list'
        self.api_workspace = f'{base_url}/api/2.0/workspace/export'

    @staticmethod
    def __extract_run_libraries(run_tags):
        libraries_objects = []
        if 'mlflow.databricks.cluster.libraries' in run_tags:
            # TODO: extract python libraries and DBFS uploads
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
    def __extract_run_cluster(tags):
        cluster_info = tags.get('mlflow.databricks.cluster.info') or None
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
                num_workers,
            )
        else:
            return None

    @staticmethod
    def __extract_data_sources_lineage(source_records):
        mlflow_data_records = []
        for source_record in source_records:
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
        return ExperimentDataSources(mlflow_data_records)

    @staticmethod
    def __extract_run_data_sources(tags):
        mlflow_data = tags.get('sparkDatasourceInfo') or None
        if mlflow_data:
            return mlflow_data.split('\n')
        else:
            return []

    @staticmethod
    def __extract_run_artifact_signature(fields):
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

    def __extract_run_artifacts(self, tags):
        model_info = tags['mlflow.log-model.history']
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
    def __extract_model_tags(model_object):
        if 'tags' in model_object:
            return key_value_to_dict(model_object['tags'])
        else:
            return {}

    @staticmethod
    def __extract_model_description(model_object):
        if 'description' in model_object:
            return ModelDescription(model_object['description'])
        else:
            return None

    @staticmethod
    def __get_latest_model_version(response):
        latest_versions = response['registered_model_databricks']['latest_versions']
        return int(sorted(latest_versions, key=lambda x: int(x['version']))[-1]['version'])

    def __process_model_submission(self, response, model_name, model_version):

        model_stage = ModelStage(response['current_stage'])
        model_date = parse_date(response['creation_timestamp'])
        model_run_id = response['run_id']
        model_owner = response['user_id']

        # Retrieve markdown text from registered model
        model_description = self.__extract_model_description(response)
        model_tags = self.__extract_model_tags(response)

        # Retrieve transition stage - if any
        model_transition = self.__get_transition(model_name, model_version)

        return ModelSubmission(
            model_description,
            model_tags,
            model_owner,
            model_version,
            model_date,
            model_stage,
            model_transition,
            model_run_id
        )

    def __process_model_parent(self, response, model_name):

        model_parent = response['registered_model_databricks']
        model_parent_owner = model_parent['user_id']
        model_parent_creation = parse_date(model_parent['creation_timestamp'])
        model_parent_description = self.__extract_model_description(model_parent)
        model_parent_tags = self.__extract_model_tags(model_parent)
        model_parent_latest_versions = model_parent['latest_versions']

        model_submissions = {}
        for revision in model_parent_latest_versions:
            revision_version = int(revision['version'])
            model_submissions[revision_version] = revision

        return ModelParent(
            model_name,
            model_parent_description,
            model_parent_tags,
            model_parent_owner,
            model_parent_creation,
            model_submissions
        )

    def __process_run(self, response, run_id):

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

        run_data_sources = self.__extract_run_data_sources(run_tags)
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
    def __extract_source_lineage_name(upstream):
        if 'tableInfo' in upstream:
            table_info = upstream['tableInfo']
            coordinate = []
            if 'catalog_name' in table_info:
                coordinate.append(table_info['catalog_name'])
            if 'schema_name' in table_info:
                coordinate.append(table_info['schema_name'])
            else:
                coordinate.append('default')
            if 'name' in table_info:
                coordinate.append(table_info['name'])
                return '.'.join(coordinate)
            else:
                logger.warning("No table name found, ignoring upstream")
        else:
            logger.warning("Unsupported format for source {}".format(upstream.keys()))
        return None

    def __process_lineage(self, response):
        children = []
        if 'upstreams' in response:
            upstreams = response['upstreams']
            for upstream in upstreams:
                upstream_source = self.__extract_source_lineage_name(upstream)
                upstream_lineage = self.__get_lineage_rec(upstream_source)
                children.append(upstream_lineage)
        return children

    @staticmethod
    def __process_notebook(html_content):
        notebook_regex = re.compile(
            "DATABRICKS_NOTEBOOK_MODEL = '((?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?)'"
        )
        matches = notebook_regex.findall(html_content)
        if matches:
            return Notebook(matches[0])
        else:
            logger.error("Could not extract notebook content from HTML")
            return None

    @staticmethod
    def __process_transition(response):
        if 'requests' in response:
            requests_response = response['requests']
            requests_response_sorted = list(sorted(requests_response, key=lambda x: x['creation_timestamp']))
            if requests_response_sorted:
                return ModelStage(requests_response_sorted[-1]['to_stage'])
            else:
                return None
        else:
            return None

    def __get_model_parent(self, model_name):
        logger.info(f'Retrieving model [{model_name}] from mlflow API')
        url = f'{self.api_registry}?name={model_name}'
        response = json.loads(requests.get(url=url, headers=self.headers).text)
        if 'error_code' in response:
            logger.error(f"Error in model response, {response['message']}")
            return None
        return self.__process_model_parent(response, model_name)

    def __get_run(self, run_id):
        logger.info(f'Retrieving run_id [{run_id}] associated to model')
        url = f'{self.api_run}?run_id={run_id}'
        response = json.loads(requests.get(url=url, headers=self.headers).text)
        if 'error_code' in response:
            logger.error(f"Error is run response, {response['message']}")
            return None
        return self.__process_run(response, run_id)

    def __get_notebook_from_job(self, job_id):
        run_id = job_id.split('/')[-1]
        logger.info(f'Retrieving notebook associated to job [{run_id}]')
        url = f'{self.api_job_export}?run_id={run_id}&views_to_export=CODE'
        response = json.loads(requests.get(url=url, headers=self.headers).text)
        if 'error_code' in response:
            logger.error(f"Error in job response, {response['message']}")
            return None
        content = list(filter(lambda x: x['type'] == 'NOTEBOOK', response['views']))
        if len(content) > 0:
            html_org_content = content[0]['content']
            return self.__process_notebook(html_org_content)
        else:
            logger.error(f"Could not find any output content for job [{job_id}]")
            return None

    def __get_transition(self, model_name, model_version):
        logger.info(f'Retrieving transition requests for model [{model_name}] version {model_version}')
        url = f'{self.api_transitions}?name={model_name}&version={model_version}'
        response = json.loads(requests.get(url=url, headers=self.headers).text)
        if 'error_code' in response:
            logger.error(f"Error in transition response, {response['message']}")
            return None
        return self.__process_transition(response)

    def __get_notebook(self, remote_path):
        logger.info(f'Retrieving notebook [{remote_path}] associated to model run')
        url = f'{self.api_workspace}?path={remote_path}&format=HTML&direct_download=False'
        response = json.loads(requests.get(url=url, headers=self.headers).text)
        if 'error_code' in response:
            logger.error(f"Error in notebook response, {response['message']}")
            return None
        html_org_content = str(base64.b64decode(response['content']), 'utf-8')
        return self.__process_notebook(html_org_content)

    def __get_lineage_rec(self, data_source_name):
        logger.info(f'Retrieving lineage for data source [{data_source_name}]')
        url = f'{self.api_lineage}?table_name={data_source_name}'
        response = json.loads(requests.get(url=url, headers=self.headers).text)
        if 'error_code' in response:
            logger.error(f"Error in lineage response, {response['message']}")
            children = []
        else:
            children = self.__process_lineage(response)
        return LineageDataSource(data_source_name, children)

    def __get_lineage(self, data_source_names):
        return Lineage([self.__get_lineage_rec(data_source_name) for data_source_name in data_source_names])

    def generate_doc(self, model_name, output_file, model_version=None):

        verbatim = load_verbatim()

        if model_version:
            logger.info(f'Generating MRM output for model [{model_name}] (v{model_version})')
        else:
            logger.info(f'Generating MRM output for latest model [{model_name}]')

        model_parent = self.__get_model_parent(model_name)
        if not model_parent:
            raise Exception(f"Could not find model {model_name} on ml registry")

        if model_version:
            if model_version in model_parent.model_submissions.keys():
                model_submission_response = model_parent.model_submissions[model_version]
                model_submission = self.__process_model_submission(model_submission_response, model_name, model_version)
            else:
                raise Exception(f'Could not find version {model_version} for model {model_name}, make sure version '
                                f'is the latest version for the given stage')
        else:
            model_latest_version = sorted(model_parent.model_submissions.keys())[-1]
            logger.info(f'Found latest version {model_latest_version} for model [{model_name}]')
            model_submission_response = model_parent.model_submissions[model_latest_version]
            model_submission = self.__process_model_submission(model_submission_response, model_name,
                                                               model_latest_version)

        model_run = self.__get_run(model_submission.model_run_id)
        if not model_run:
            raise Exception(f"Could not find experiment {model_submission.model_run_id} "
                            f"associated with model {model_name}")

        if model_run.run_data_sources:
            data_sources = self.__extract_data_sources_lineage(model_run.run_data_sources)
        else:
            data_sources = None

        if data_sources:
            data_lineage = self.__get_lineage(data_sources.sources())
        else:
            data_lineage = None

        ##########################################################################################
        ##########################################################################################

        html = [
            f'<h1 class="text-center">{verbatim["title"]}</h1>',
            f'<h3 class="card-subtitle text-muted text-center">{model_name}</h3>',
        ]

        ##########################################################################################
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
        ##########################################################################################

        html.extend([
            '<div class="section section-break section-content">',
            f'<h1>{verbatim["mlflow_model_version"]["header"]}</h1>',
            f'<p>{verbatim["mlflow_model_version"]["info"]}</p>',
            a_bit_of_space
        ])

        html.extend(model_submission.to_html(h_level=1))
        html.append(a_little_bit_of_space)
        if model_submission.model_description:
            html.append('<small class="text-muted">description from mlflow registry</small>')
            html.extend(model_submission.model_description.to_html(h_level=2))
        else:
            html.extend([
                '<div class="alert alert-warning section-content" role="alert">',
                f'<p>{verbatim["mlflow_model_version"]["error"]}</p>',
                '</div>'
            ])

        ##########################################################################################
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
        ##########################################################################################

        html.extend([
            '<div class="section section-break section-content">',
            f'<h1>{verbatim["implementation"]["header"]}</h1>',
            f'<p>{verbatim["implementation"]["info"]}</p>',
            '</div>'
        ])

        ##########################################################################################
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
        ##########################################################################################

        html.extend([
            '<div class="section section-break section-content">',
            f'<h1>{verbatim["model_dependencies"]["header"]}</h1>',
            f'<p>{verbatim["model_dependencies"]["info"]}</p>',
            '</div>'
        ])

        ##########################################################################################
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
        ##########################################################################################

        generate_pdf(html, output_file)
