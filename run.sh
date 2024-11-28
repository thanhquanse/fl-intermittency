#!/bin/sh

# notebook_arr="FL-DT-full-client-updates-weather.ipynb FL-DT-no-updates-ignored-10x10-01-noadjacency-weather.ipynb"
# notebook_arr="FL-DT-full-client-updates-electricity.ipynb FL-DT-full-client-updates-traffic.ipynb FL-DT-full-client-updates-psm.ipynb FL-DT-full-client-updates-weather.ipynb"
# notebook_arr="FL-DT-no-updates-ignored-10x10-01-noadjacency-electricity.ipynb FL-DT-no-updates-ignored-10x10-01-noadjacency-weather.ipynb FL-DT-no-updates-ignored-10x10-01-noadjacency-traffic.ipynb FL-DT-no-updates-ignored-10x10-01-noadjacency-psm.ipynb"
# notebook_arr="FL-DT-no-updates-ignored-10x10-02-2adjacency-electricity.ipynb FL-DT-no-updates-ignored-10x10-02-2adjacency-weather.ipynb FL-DT-no-updates-ignored-10x10-02-2adjacency-traffic.ipynb FL-DT-no-updates-ignored-10x10-02-2adjacency-psm.ipynb"
# notebook_arr="FL-DT-proposed-method-10x10-01-noadjacency-electricity.ipynb FL-DT-proposed-method-10x10-01-noadjacency-weather.ipynb FL-DT-proposed-method-10x10-01-noadjacency-traffic.ipynb FL-DT-proposed-method-10x10-01-noadjacency-psm.ipynb"

# for notebook in $notebook_arr; do
#     echo "Processing $notebook..."
#     jupyter nbconvert --execute --to notebook --allow-errors $notebook
#     echo "Finished processing $notebook"
#     echo
#     echo
#     echo
# done

DATASET="electricity weather traffic psm"
METHODS="normal avg weight"
PERCENT_MC="01 02 03 04 05"
MISSING_MODE="noadjacency 2adjacency 3adjacency 4adjacency 5adjacency"
WEIGHT_MECHANISM="0 1"

for ds in $DATASET; do
    for mt in $METHODS; do
        if [ "$mt" != "normal" ]; then
            for pmc in $PERCENT_MC; do
                for mm in $MISSING_MODE; do
                    for wm in $WEIGHT_MECHANISM; do
                        echo "Processing $ds - $mt - $pmc - $mm - $wm..."
                        python run.py --dataset $ds --prefix $mt --percent_mc $pmc --missing_mode $mm --weight_mechanism $wm
                        echo "Done"
                    done
                done
            done
        else
            echo "Processing $ds - $mt..."
            python run.py --dataset $ds --prefix $mt
            echo "Done"
        fi
    done
done

# cli="run.py '--dataset', 'electricity', '--prefix', 'avg', '--percent_mc', '01', '--missing_mode', 'noadjacency' '--weight-mechanism'"

# --dataset electricity --prefix avg --percent_mc 01 --missing_mode noadjacency --weight-mechanism 0