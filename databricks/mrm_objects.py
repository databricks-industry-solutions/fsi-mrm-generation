import abc
import base64
import json
from urllib.parse import unquote

from graphviz import Digraph

from databricks.mrm_utils import *


class MRMInterface:
    @abc.abstractmethod
    def to_html(self, h_level=1, dot=None):
        pass


class ModelStage(MRMInterface):
    def __init__(self, stage):
        self.stage = stage.upper()

    def to_html(self, h_level=1, dot=None):
        if self.stage == 'STAGING':
            badge = 'badge-warning'
        elif self.stage == 'ARCHIVED':
            badge = 'badge-secondary'
        elif self.stage == 'PRODUCTION':
            badge = 'badge-danger'
        else:
            badge = 'badge-secondary'
        return f'<span class="badge {badge}">{self.stage}</span>'


class ModelExperiment(MRMInterface):
    def __init__(
            self,
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
    ):
        self.run_id = run_id
        self.run_description = run_description
        self.run_user = run_user
        self.run_workspace = run_workspace
        self.run_experiment_id = run_experiment_id
        self.run_timestamp = run_timestamp
        self.run_params = run_params
        self.run_metrics = run_metrics
        self.run_data_sources = run_data_sources
        self.run_parent_id = run_parent_id
        self.source_type = source_type
        self.source_name = source_name
        self.source_url = source_url
        self.run_cluster = run_cluster
        self.run_artifacts = run_artifacts
        self.source_code = source_code
        self.source_commit = source_commit
        self.run_libraries = run_libraries

    def to_html(self, h_level=1, dot=None):
        return [
            '<div class="section-content">',
            '<table class="table">',
            '<tr>',
            '<th>Execution time</th>',
            f'<td>{self.run_timestamp}</td>',
            '</tr>',
            '<tr>'
            '<th>Execution user</th>',
            f'<td>{self.run_user}</td>',
            '</tr>',
            '<tr>'
            '<th>Execution workspace</th>',
            f'<td>{self.run_workspace}</td>',
            '</tr>',
            '<tr>',
            '<th>Execution type</th>',
            f'<td><span class="badge badge-secondary">{self.source_type}</span></td>',
            '</tr>',
            '<tr>',
            '<th>Execution code</th>',
            f'<td>{self.source_name}</td>',
            '</tr>',
            '<tr>',
            '<th>Execution code url</th>',
            f'<td>{self.source_url}</td>',
            '</tr>',
            '<tr>',
            '<th>Execution code revision</th>',
            f'<td>{self.source_commit}</td>',
            '</tr>',
            '</table>',
            '</div>'
        ]


class ExperimentMetrics(MRMInterface):
    def __init__(
            self,
            metrics
    ):
        self.metrics = metrics

    def to_html(self, h_level=1, dot=None):

        html = []
        html.extend([
            '<div class="section-content">',
            '<table class="table">',
            '<tr>',
            '<th>Metric</th>',
            '<th>Value</th>',
            '</tr>',
        ])
        for key in self.metrics.keys():
            if self.metrics[key] != 'None':
                html.extend([
                    '<tr>',
                    f'<td>{key}</td>',
                    f'<td>{self.metrics[key]}</td>',
                    '</tr>'
                ])
        html.extend([
            '</table>',
            '</div>'
        ])
        return html


class ExperimentParameters(MRMInterface):
    def __init__(
            self,
            parameters
    ):
        self.parameters = parameters

    def to_html(self, h_level=1, dot=None):
        html = []
        html.extend([
            '<div class="section-content">',
            '<table class="table">',
            '<tr>',
            '<th>Parameter</th>',
            '<th>Value</th>',
            '</tr>',
        ])
        for key in self.parameters.keys():
            if self.parameters[key] != 'None':
                html.extend([
                    '<tr>',
                    f'<td>{key}</td>',
                    f'<td>{self.parameters[key]}</td>',
                    '</tr>'
                ])
        html.extend([
            '</table>',
            '</div>'
        ])
        return html


class ExperimentLoggedModel(MRMInterface):

    def __init__(
            self,
            inputs,
            outputs,
            flavor
    ):
        self.inputs = inputs
        self.outputs = outputs
        self.flavor = flavor

    def to_html(self, h_level=1, dot=None):
        raise Exception("Unsupported")


class ExperimentCluster(MRMInterface):
    def __init__(
            self,
            name,
            dbr,
            cloud_instance,
            num_workers
    ):
        self.dbr = dbr
        self.name = name
        self.cloud_instance = cloud_instance
        self.num_workers = num_workers

    def to_html(self, h_level=1, dot=None):
        return [
            '<div class="section-content">',
            '<table class="table">',
            '<tr>',
            '<th>Cluster name</th>',
            f'<td>{self.name}</td>',
            '</tr>',
            '<tr>',
            '<th>Cluster runtime</th>',
            f'<td>{self.dbr}</td>',
            '</tr>',
            '<tr>'
            '<th>Cluster instance type</th>',
            f'<td>{self.cloud_instance}</td>',
            '</tr>',
            '<tr>',
            '<th>Cluster number workers</th>',
            f'<td>{self.num_workers}</td>',
            '</tr>',
            '</table>',
            '</div>'
        ]


class ExperimentDescription(MRMInterface):
    def __init__(
            self,
            description
    ):
        self.description = description

    def to_html(self, h_level=1, dot=None):
        html = ['<div class="section-content">']
        html.extend(markdown_to_html(self.description, h_level=h_level, container=True))
        html.append('</div>')
        return html


class ExperimentDataSource(MRMInterface):
    def __init__(
            self,
            name,
            fmt,
            version
    ):
        self.name = name
        self.fmt = fmt
        self.version = version

    def to_html(self, h_level=1, dot=None):
        return [
            '<tr>',
            f'<td>{self.name}</td>',
            f'<td>{self.fmt}</td>',
            f'<td>{self.version}</td>',
            '</tr>'
        ]


class ExperimentDataSources(MRMInterface):
    def __init__(
            self,
            data_sources
    ):
        self.data_sources = data_sources

    def sources(self):
        return [[source.name, source.fmt] for source in self.data_sources]

    def to_html(self, h_level=1, dot=None):
        html = [
            '<div class="section-content">',
            '<table class="table">',
            '<tr>',
            '<th>Path</th>',
            '<th>Format</th>',
            '<th>Version</th>',
            '</tr>'
        ]
        for data_source in self.data_sources:
            html.extend(data_source.to_html(h_level))
        html.extend([
            '</table>',
            '</div>'
        ])
        return html


class Notebook(MRMInterface):

    def __init__(
            self,
            content
    ):
        self.content = content

    def to_html(self, h_level=1, dot=None):
        org_notebook = base64.b64decode(self.content).decode('utf-8')
        org_notebook = json.loads(unquote(org_notebook))
        html_body = ['<div class="section">']
        for i, command in enumerate(org_notebook['commands']):
            command_value = command['command']
            if command_value.startswith('%md'):
                html_body.append('<div>')
                html_body.append(f'<small class="text-muted">markdown cell #{i}</small>')
                html_body.append('</div>')
                html_body.append('<div class="container jumbotron section-markdown">')
                html_body.extend(markdown_to_html(command_value, h_level))
                html_body.append('</div>')
            else:
                if command['results']:
                    cmd_output = command['results']
                    data_list = cmd_output['data']
                    for data_entry in data_list:
                        if data_entry['type'] == 'mimeBundle':
                            bundle = data_entry['data']
                            if 'image/png' in bundle:
                                image_b64 = bundle['image/png']
                                html_body.append('<div>')
                                html_body.append(f'<small class="text-muted">output cell #{i}</small>')
                                html_body.append('</div>')
                                html_body.append('<div class="container jumbotron section-markdown">')
                                html_body.extend(image_to_html(image_b64))
                                html_body.append('</div>')
        html_body.append('</div>')
        return html_body


class ModelPreviousVersion(MRMInterface):
    def __init__(
            self,
            version,
            timestamp,
            user,
            alias
    ):
        self.version = version
        self.timestamp = timestamp
        self.user = user
        self.alias = alias

    def to_html(self, h_level=1, dot=None):
        html = [
                '<tr>',
                f'<td>{self.version}</td>',
                f'<td>{self.timestamp}</td>',
                f'<td>{self.user}</td>'
            ]

        if self.alias:
            html.append(f'<td><span class="badge badge-secondary">@{self.alias}</span></td>')
        else:
            html.append(f'<td></td>')
        html.append('</tr>')
        return html


class ModelVersions(MRMInterface):
    def __init__(
            self,
            model_name,
            model_versions
    ):
        self.model_name = model_name
        self.model_versions = model_versions

    def to_html(self, h_level=1, dot=None):

        html = [
            '<div class="section-content">',
            '<table class="table">',
            '<tr>',
            '<th>Version</th>',
            '<th>Timestamp</th>',
            '<th>Owner</th>',
            '<th>Alias</th>',
            '</tr>'
        ]

        for version in sorted(self.model_versions, key=lambda x: x.version):
            html.extend(version.to_html())

        html.extend(['</table>', '</div>'])
        return html


class ModelParent(MRMInterface):
    def __init__(
            self,
            model_catalog,
            model_schema,
            model_name,
            model_description,
            model_tags,
            model_owner,
            model_timestamp,
            model_parent_aliases
    ):
        self.model_catalog = model_catalog
        self.model_schema = model_schema
        self.model_name = model_name
        self.model_description = model_description
        self.model_tags = model_tags
        self.model_owner = model_owner
        self.model_timestamp = model_timestamp
        self.model_parent_aliases = model_parent_aliases

    def to_html(self, h_level=1, dot=None):
        html = [
            '<div class="section-content">',
            '<table class="table">',
            '<tr>',
            '<th>Model name</th>',
            f'<td>{self.model_name}</td>',
            '</tr>',
            '<tr>',
            '<th>Model catalog</th>',
            f'<td>{self.model_catalog}</td>',
            '</tr>',
            '<tr>',
            '<th>Model schema</th>',
            f'<td>{self.model_schema}</td>',
            '</tr>',
            '<tr>',
            '<th>Model creation date</th>',
            f'<td>{self.model_timestamp}</td>',
            '</tr>',
            '<tr>',
            '<th>Model owner</th>',
            f'<td>{self.model_owner}</td>',
            '</tr>',
        ]

        for key in self.model_tags:
            tag = ' '.join(key.split('_')).capitalize()
            html.extend([
                '<tr>',
                f'<th><i class="bi bi-tag-fill"></i> {tag}</th>',
                f'<td>{self.model_tags[key]}</td>',
                '</tr>'
            ])

        html.extend(['</table>', '</div>'])
        return html


class ModelSubmission(MRMInterface):
    def __init__(
            self,
            model_description,
            model_owner,
            model_version,
            model_timestamp,
            model_run_id,
            model_tags
    ):
        self.model_description = model_description
        self.model_owner = model_owner
        self.model_version = model_version
        self.model_timestamp = model_timestamp
        self.model_run_id = model_run_id
        self.model_tags = model_tags

    def to_html(self, h_level=1, dot=None):
        html = [
            '<div class="section-content">',
            '<table class="table">',
            '<tr>',
            '<th>Model submission date</th>',
            f'<td>{self.model_timestamp}</td>',
            '</tr>',
            '<tr>',
            '<th>Model version owner</th>',
            f'<td>{self.model_owner}</td>',
            '</tr>',
            '<tr>',
            '<th>Model version</th>',
            f'<td>{self.model_version}</td>',
            '</tr>',
            '<tr>',
            '<th>Model run Id</th>',
            f'<td>{self.model_run_id}</td>',
            '</tr>',
        ]

        for key in self.model_tags:
            tag = ' '.join(key.split('_')).capitalize()
            html.extend([
                '<tr>',
                f'<th><i class="bi bi-tag-fill"></i> {tag}</th>',
                f'<td>{self.model_tags[key]}</td>',
                '</tr>'
            ])

        html.extend(['</table>', '</div>'])
        return html


class ModelDescription(MRMInterface):
    def __init__(
            self,
            description
    ):
        self.description = description

    def to_html(self, h_level=1, dot=None):
        html = ['<div class="section-content">']
        html.extend(markdown_to_html(self.description, h_level=h_level, container=True))
        html.append('</div>')
        return html


class Artifacts(MRMInterface):
    def __init__(
            self,
            artifacts
    ):
        self.artifacts = artifacts

    def to_html(self, h_level=1, dot=None):
        html = [
            '<div class="section-content">',
            '<table class="table">',
            '<tr>',
            '<th>Logged time</th>',
            '<th>Artifact</th>',
            '<th>Interpreter version</th>',
            '</tr>'
        ]
        for artifact in self.artifacts:
            html.extend(artifact.to_html(h_level=h_level))
        html.extend([
            '</table>',
            '</div>'
        ])
        return html


class Artifact(MRMInterface):
    def __init__(
            self,
            created,
            flavors,
            signature
    ):
        self.created = created
        self.flavors = flavors
        self.signature = signature

    def to_html(self, h_level=1, dot=None):
        html = []
        for flavor in self.flavors:
            html.extend(['<tr>', f'<td>{self.created}</td>'])
            html.extend(flavor.to_html(h_level))
            html.append('</tr>')
        return html


class ArtifactFlavor(MRMInterface):
    def __init__(
            self,
            flavor_type,
            flavor_version
    ):
        self.flavor_type = flavor_type
        self.flavor_version = flavor_version

    def to_html(self, h_level=1, dot=None):
        return [
            f'<td><span class="badge badge-secondary">{self.flavor_type}</span></td>'
            f'<td>{self.flavor_version}</td>'
        ]


class ArtifactSignature(MRMInterface):
    def __init__(
            self,
            inputs,
            outputs
    ):
        self.inputs = inputs
        self.outputs = outputs

    def to_html(self, h_level=1, dot=None):
        dot = Digraph(comment='model signature', format='png', graph_attr={'rankdir': 'LR', 'size': '7.75,10.25'})
        dot.node(str(0), label='MODEL', color='black', shape='circle', fontname='courier')
        for i, field in enumerate(self.inputs):
            dot.node(str(i + 1), label=field, color='black', shape='box', fontname='courier')
            dot.edge(str(i + 1), str(0))
        for i, field in enumerate(self.outputs):
            dot.node(str(i + 1 + len(self.inputs)), label=field, color='black', shape='box', fontname='courier')
            dot.edge(str(0), str(i + 1 + len(self.inputs)))
        b64_img = base64.b64encode(dot.pipe()).decode('ascii')
        return [
            '<div class="section-content">',
            f'<img src="data:image/png;base64, {b64_img}"/>',
            '<div class="col-xs-12" style="height:20px;"></div>',
            '</div>'
        ]


class Library(MRMInterface):
    def __init__(
            self,
            repository,
            artifact
    ):
        self.repository = repository
        self.artifact = artifact

    def to_html(self, h_level=1, dot=None):
        return [
            '<tr>',
            f'<td>{self.repository}</td>',
            f'<td>{self.artifact}</td>',
            '</tr>'
        ]


class Libraries(MRMInterface):
    def __init__(
            self,
            libraries
    ):
        self.libraries = libraries

    def non_empty(self):
        if self.libraries and len(self.libraries) > 0:
            return True
        else:
            return False

    def to_html(self, h_level=1, dot=None):
        html = [
            '<div class="section-content">',
            '<table class="table">',
            '<tr>',
            '<th>Repository</th>',
            '<th>Library</th>',
            '</tr>'
        ]
        for library in self.libraries:
            html.extend(library.to_html(h_level=h_level))
        html.extend([
            '</table>',
            '</div>'
        ])
        return html


class Lineage(MRMInterface):
    def __init__(
            self,
            data_sources,
            model_name
    ):
        self.data_sources = data_sources
        self.model_name = model_name

    def to_html(self, h_level=1, dot=None):
        dot = Digraph(comment='model lineage', format='png', graph_attr={'rankdir': 'LR', 'size': '7.75,10.25'})
        dot.node('MODEL', label=f'<<model>>\n{self.model_name}', color='black', shape='box', fontname='courier')
        for data_source in self.data_sources:
            dot.edge(string_to_uid(data_source.short_name()), 'MODEL')
            data_source.to_html(dot=dot)
        b64_img = base64.b64encode(dot.pipe()).decode('ascii')
        return [
            '<div class="section-content">',
            f'<img src="data:image/png;base64, {b64_img}" width="1150"/>',
            '<div class="col-xs-12" style="height:20px;"></div>',
            '</div>'
        ]


class LineageDataSource(MRMInterface):
    def __init__(
            self,
            name,
            type,
            children
    ):
        self.name = name
        self.type = type
        self.children = children

    def short_name(self):
        return self.name.split('/')[-1]

    def to_html(self, h_level=1, dot=None):
        node_id = string_to_uid(self.name)
        dot.node(node_id, label=f'<<{self.type}>>\n{self.name}', color='black', shape='box', fontname='courier')
        for child in self.children:
            dot.edge(string_to_uid(child.name), node_id)
            child.to_html(dot=dot)
