#!/bin/bash


# Get the latest version
VERSION=$(<VERSION)

# Tell sphinx-apidoc to look for packages in the current directory. We have to
# say "../${curr_dir_name}" instead of "." because sphinx gets the top-level
# project name from the directory name (if we used the latter, the generated
# documentation would call the top-level project "." in the table of contents)
#
# Also tell sphinx-apidoc to ignore the manage.py and setup.py files
curr_dir_name=${PWD##*/}
rm -rf docs/source/api
sphinx-apidoc -f -e -o docs/source/api ../${curr_dir_name}/ manage.py setup.py

# Now generate the html docs
python setup.py build_sphinx

