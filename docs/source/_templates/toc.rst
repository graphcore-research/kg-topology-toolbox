..
   # Copyright (c) 2023 Graphcore Ltd. All rights reserved.
   # Copyright (c) 2007-2023 by the Sphinx team. All rights reserved.

{{ header | heading }}

.. toctree::
   :maxdepth: {{ maxdepth }}
{% for docname in docnames %}
   {{ docname }}
{%- endfor %}

