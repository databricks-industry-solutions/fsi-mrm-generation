<img src=https://d1r5llqwmkrl74.cloudfront.net/notebooks/fsi/fs-lakehouse-logo-transparent.png width="600px">

# Model risk management

**How to build models that move quickly through validation and audit**: *With regulators and policymakers seriously
addressing the challenges of AI in finance and banks starting to demand more measurable profits around the use of data,
data practices are forced to step up their game in the delivery of ML if they want to drive competitive insights
reliable enough for business to trust and act upon. This utility library will automate the generation of PDF report for
submitted new models as part of a model risk management practice. Using mlflow, delta lake, unity catalog and their
respective APIs, we aim at bringing both the technical and business context surrounding your model submission, reducing
time to market and facilitating independent validation of financial and non-financial models.*

[![DBR](https://img.shields.io/badge/DBR-11.3ML-red?logo=databricks&style=for-the-badge)](https://docs.databricks.com/release-notes/runtime/11.3ml.html)
[![CLOUD](https://img.shields.io/badge/CLOUD-ALL-blue?logo=googlecloud&style=for-the-badge)](https://databricks.com/try-databricks)
[![POC](https://img.shields.io/badge/POC-1_days-green?style=for-the-badge)](https://databricks.com/try-databricks)

___

<antoine.amend@databricks.com>

___

## Usage

```python
from databricks.mrm import ModelRiskApi

mrm = ModelRiskApi(
    databricks_workspace_url,
    databricks_token
)

mrm.generate_doc(
    model_name=model_name, # name of the model on ML registry
    model_version=model_version, # version of the model (optional, default is latest)
    output_file=output_file # name of the output file for PDF document
)
```

See example [output](templates%2FCredit%20Adjudication%20-%20Output.pdf) for a given test model done using an existing solution 
[accelerator](https://github.com/databricks-industry-solutions/value-at-risk). 

Make sure to have both `wkhtmltopdf` and `graphviz` binary installed

```bash
sudo apt-get install -y graphviz wkhtmltopdf
```

### Command line

Should you need to run the same from a command line utility, please refer to `databricks.py`.

```shell
pip install -r requirements.txt
python databricks.py \
    --db-workspace my-workspace-url \
    --db-token my-workspace-token \
    --model-name my-model-name \
    --model-version my-model-version \
    --output my-model-output.pdf
```

## Template documentation

This utility library provides a technical foundation and framework to automatically 
generate the document and represents one side of the coin. The other side relates to the 
design, the structure and the content of the document that relates to the model in scope. 
With subject matter expertise supporting several model frameworks, regulator guidelines and 
custom needs of many FS institutions, the 
[MRM suite](https://www.ey.com/en_gl/financial-services/model-management-platform)
from EY provides proven model documentation templates corresponding to business domains 
in the financial services industry.

See an example of [template](templates/Credit%20Adjudication%20-%20MRM%20Model%20Documentation%20-%20Template.ipynb) 
document for Credit Risk Adjudication use case. Available as a Ipython notebook file, hence easily
accessible through Databricks environment, this template provides the necessary placemat through 
markdown comments to generate the documentation required for Model Risk Management.

## License

© 2023 Databricks, Inc. All rights reserved. The source is provided subject to the Databricks License
[https://databricks.com/db-license-source]. All included or referenced third party libraries are subject to the licenses
set forth below.

| library                          | description         | license | source                                     |
|----------------------------------|---------------------|---------|--------------------------------------------|
| PyYAML                           | Yaml parser         | MIT     | https://pypi.org/project/PyYAML/           |
| mdtex2html                       | Markdown parser     | MIT     | https://pypi.org/project/mdtex2html/        |
| graphviz                         | Graph visualization | MIT     | https://pypi.org/project/graphviz/         |
| pdfkit                           | html to pdf         | MIT     | https://pypi.org/project/pdfkit/           |



