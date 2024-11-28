#!/bin/sh

# notebook_arr="FL-DT-full-client-updates-weather.ipynb FL-DT-no-updates-ignored-10x10-01-noadjacency-weather.ipynb"
# notebook_arr="FL-DT-full-client-updates-electricity.ipynb FL-DT-full-client-updates-traffic.ipynb FL-DT-full-client-updates-psm.ipynb FL-DT-full-client-updates-weather.ipynb"
notebook_arr="FL-DT-no-updates-ignored-10x10-01-noadjacency-electricity.ipynb FL-DT-no-updates-ignored-10x10-01-noadjacency-weather.ipynb FL-DT-no-updates-ignored-10x10-01-noadjacency-traffic.ipynb FL-DT-no-updates-ignored-10x10-01-noadjacency-psm.ipynb"
# notebook_arr="FL-DT-no-updates-ignored-10x10-02-2adjacency-electricity.ipynb FL-DT-no-updates-ignored-10x10-02-2adjacency-weather.ipynb FL-DT-no-updates-ignored-10x10-02-2adjacency-traffic.ipynb FL-DT-no-updates-ignored-10x10-02-2adjacency-psm.ipynb"
# notebook_arr="FL-DT-proposed-method-10x10-01-noadjacency-electricity.ipynb FL-DT-proposed-method-10x10-01-noadjacency-weather.ipynb FL-DT-proposed-method-10x10-01-noadjacency-traffic.ipynb FL-DT-proposed-method-10x10-01-noadjacency-psm.ipynb"

for notebook in $notebook_arr; do
    echo "Processing $notebook..."
    jupyter nbconvert --execute --to notebook --allow-errors $notebook
    echo "Finished processing $notebook"
    echo
    echo
    echo
done
    