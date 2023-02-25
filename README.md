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

api = ModelRiskApi(
    databricks_workspace_url, 
    databricks_token
)

api.generate_mrm(
    model_name, 
    pdf_output_file, 
    model_version=model_version
)
```

Make sure to have both wkhtmltopdf and graphviz binary installed

```bash
sudo apt-get install -y graphviz wkhtmltopdf
```

## License

© 2023 Databricks, Inc. All rights reserved. The source is provided subject to the Databricks License
[https://databricks.com/db-license-source]. All included or referenced third party libraries are subject to the licenses
set forth below.

| library                          | description         | license | source                                     |
|----------------------------------|---------------------|---------|--------------------------------------------|
| PyYAML                           | Yaml parser         | MIT     | https://pypi.org/project/PyYAML/           |
| markdown                         | Markdown parser     | MIT     | https://pypi.org/project/markdown2/        |
| graphviz                         | Graph visualization | MIT     | https://pypi.org/project/graphviz/         |
| pdfkit                           | html to pdf         | MIT     | https://pypi.org/project/pdfkit/           |


*Please note that this utility library was built as a framework rather than as an end product. As generic as possible,
this framework may not accommodate every specific requirements across different model strategies, different model
materiality and different organizations' policies. We highly encourage practitioners to build upon this framework to
cover their organizations' internal policies. For policies not programmatically covered by the use of databricks
notebooks, clusters, mlflow or unity catalog, it becomes the responsibility of industry practitioners to complement the
same through the use of additional markdown documentation. We highly recommend organizations and their compliance teams
to standardize a series of controls as a form of
'template notebooks' data team can follow as part of their development practices.*