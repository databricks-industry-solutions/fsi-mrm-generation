import datetime
import importlib.resources as pkg_resources
import os
import re
import shutil
import tempfile

import mdtex2html
import pdfkit
import yaml

from . import tmpl


def load_verbatim(verbatim_file=None):
    """
    We do not wish to hardcode all text for our given PDF output but rather enable users to override their own
    narrative.
    :param verbatim_file: optional path for YAML file including all desired narrative
    :return: parse YAML object that will be pulled in PDF content
    """
    if verbatim_file:
        with open(verbatim_file, 'r') as f:
            return yaml.safe_load(f.read())
    else:
        return yaml.safe_load(pkg_resources.read_text(tmpl, 'verbatim.yml'))


def extract_aliases(model_object):
    """
    Utility function converting MLFlow alias object to key value pairs
    :param model_object:
    :return:
    """
    alias_kv = {}
    if 'aliases' in model_object:
        aliases = model_object['aliases']
        for alias in aliases:
            alias_kv[int(alias['version'])] = alias['alias']
    return alias_kv


def extract_tags(model_object):
    """
    Utility function converting MLFlow tag object to key value pairs
    :param model_object:
    :return:
    """
    if 'tags' in model_object:
        return key_value_to_dict(model_object['tags'])
    else:
        return {}


def generate_pdf(html_input, output_file):
    """
    Utility function that will convert raw HTML to glossy PDF. To do so, we load HTML template and relevant CSS file,
    copy all assets to a temporary folder and convert the whole as well defined PDF output. The latter will leverage
    pdfkit library and its native binary (that needs to be installed)
    :param html_input: input RAW HTML file for our model
    :param output_file: the output file to write PDF to
    :return: None
    """
    # Create HTML file
    with pkg_resources.path(tmpl, 'mrm.html') as p:
        html_output = p.read_text().format(html_body='\n'.join(html_input))

    # Create HTML folder
    temp_dir = tempfile.TemporaryDirectory()

    # Copy HTML file
    html_output_file = os.path.join(temp_dir.name, 'model.html')
    with open(html_output_file, 'w') as f:
        f.write(html_output)

    # Copy HTML assets
    asset_dir = os.path.join(temp_dir.name, 'assets')
    asset_src_dir = os.path.join(os.path.dirname(tmpl.__file__), 'assets')
    shutil.copytree(asset_src_dir, asset_dir)

    # loading header file
    with pkg_resources.path(tmpl, 'header.html') as f:
        dst_header = os.path.join(os.getcwd(), f)

    # loading footer file
    with pkg_resources.path(tmpl, 'footer.html') as f:
        dst_footer = os.path.join(os.getcwd(), f)

    # formatting options
    options = {
        'enable-local-file-access': None,
        '--header-html': 'file://{}'.format(dst_header),
        '--footer-html': 'file://{}'.format(dst_footer),
        'margin-top': '1in',
        'margin-bottom': '1in',
        'margin-right': '1in',
        'margin-left': '1in'
    }

    # generate pdf
    pdfkit.from_file(html_output_file, output_file, verbose=True, options=options)


def image_to_html(data_entry):
    """
    Convert an image from a notebook or description into bootstrap compatible HTML input
    :param data_entry: the encoded based 64 image
    :return: well formatted HTML output
    """
    return [
        '<div>',
        '<figure class="image">',
        '<img src=\'data:image/png;base64, {}\'/>'.format(data_entry),
        '</figure>',
        '</div>'
    ]


def demote_markdown(md, h_level):
    """
    Markdown can easily be converted to HTML using markdown library. Even better, markdown itself supports HTML.
    However, markdown may contain title that will conflict with our original HTML header. We therefore need to
    "demote" title to lower heading if needed.
    :param md: the original markdown file
    :param h_level: the desired level to start our heading indexing
    :return: demoted markdown
    """
    return re.sub('#\\s', '#' * (h_level + 1) + ' ', md)


def markdown_to_html(command_value, h_level=1, container=False):
    """
    Markdown can easily be converted to HTML using markdown library. Even better, markdown itself supports HTML.
    However, markdown may contain title that will conflict with our original HTML header. We therefore need to
    "demote" title to lower heading if needed.
    :param command_value: the original markdown value
    :param h_level: the desired level to start our heading indexing
    :param container: a boolean indicating the type of HTML output we need, a container or simple description
    :return: well formatted HTML code including markdown information
    """
    text = re.sub('%md\n*', '', command_value)
    text = demote_markdown(text, h_level)
    if container:
        return [f'<div class="container jumbotron">', mdtex2html.convert(text), '</div>']
    else:
        return [f'<div>', mdtex2html.convert(text), '</div>']


def key_value_to_dict(kv_pairs):
    kv_dict = {}
    for kv in kv_pairs:
        kv_dict[kv['key']] = kv['value']
    return kv_dict


def parse_date(epoch):
    dt = datetime.datetime.fromtimestamp(epoch / 1000)
    return dt.strftime('%Y-%m-%d')


def parse_time(epoch):
    dt = datetime.datetime.fromtimestamp(epoch / 1000)
    return dt.isoformat()


def string_to_uid(string):
    return ''.join(hex(ord(x))[2:] for x in string)
