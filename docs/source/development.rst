Development
===========

Local setup
-----------

Clone the repository and install it in editable mode:

.. code-block:: bash

   git clone https://github.com/Neuro-Mechatronics-Interfaces/python-open-ephys.git
   cd python-open-ephys
   pip install -e .

Validation commands
-------------------

Run the automated tests:

.. code-block:: bash

   pytest tests/

Build the package distributions:

.. code-block:: bash

   python -m build

Build the documentation site:

.. code-block:: bash

   python -m sphinx -b html docs/source docs/build/html

Release flow
------------

- Pushes to ``main`` deploy documentation to GitHub Pages.
- The TestPyPI workflow can be run manually for pre-release validation.
- Publishing a GitHub Release triggers the PyPI publish workflow.