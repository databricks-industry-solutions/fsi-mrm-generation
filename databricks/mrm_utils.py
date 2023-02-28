import datetime
import importlib.resources as pkg_resources
import os
import re
import shutil
import tempfile

import markdown
import pdfkit
import yaml

from . import tmpl


def load_verbatim():
    return yaml.safe_load(pkg_resources.read_text(tmpl, 'verbatim.yml'))


def generate_pdf(html_input, output_file):

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
    with pkg_resources.path(tmpl, 'assets') as f:
        shutil.copytree(f, asset_dir)

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
    return [
        '<div>',
        '<figure class="image">',
        '<img src=\'{}\'/>'.format(data_entry['data']),
        '</figure>',
        '</div>'
    ]


def demote_markdown(md, h_level):
    return re.sub('#\\s', '#' * (h_level + 1) + ' ', md)


def markdown_to_html(command_value, h_level=1, container=False):
    text = re.sub('%md\n*', '', command_value)
    text = demote_markdown(text, h_level)
    if container:
        return [f'<div class="container jumbotron">', markdown.markdown(text), '</div>']
    else:
        return [f'<div>', markdown.markdown(text), '</div>']


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
