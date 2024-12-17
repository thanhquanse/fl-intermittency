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

# DATASET="electricity traffic solarpower m4 train psm"
# METHODS="normal avg weight"
# PERCENT_MC="01 02 03 04 05"
# MISSING_MODE="noadjacency 2adjacency 3adjacency 4adjacency 5adjacency"
# WEIGHT_MECHANISM="1"

DATASET="electricity traffic"
METHODS="normal avg weight"
PERCENT_MC="1 2 3 4 5"
MISSING_MODE="noadjacency 2adjacency 3adjacency 4adjacency 5adjacency"
WEIGHT_MECHANISM="1"
IS_CLUSTER="no"
MATRIX="20x10"

for ds in $DATASET; do
    for mt in $METHODS; do
        if [ "$mt" != "normal" ]; then
            for pmc in $PERCENT_MC; do
                pm="0""$pmc"
                if [ "$pmc" = "1" ]; then
                    mm="noadjacency"
                else
                    mm="$pmc""adjacency"
                fi

                echo "Processing $IS_CLUSTER $ds - $mt - $pm - $mm - $MATRIX..."
                python run.py --is_cluster $IS_CLUSTER --dataset $ds --prefix $mt --percent_mc $pm --missing_mode $mm --weight_mechanism $WEIGHT_MECHANISM --matrix_ml $MATRIX
                echo "Done"
                echo
                echo
                echo
                # for mm in $MISSING_MODE; do
                #     if [ "$mt" = "avg" ]; then
                #         WEIGHT_MECHANISM="0"
                #     fi
                #     echo "Processing $ds - $mt - $pmc - $mm - $WEIGHT_MECHANISM..."
                #     python run.py --dataset $ds --prefix $mt --percent_mc $pmc --missing_mode $mm --weight_mechanism $WEIGHT_MECHANISM
                #     echo "Done"
                #     echo
                #     echo
                #     echo
                # done
            done
        else
            echo "Processing $IS_CLUSTER $ds - $mt - $MATRIX..."
            python run.py --is_cluster $IS_CLUSTER --dataset $ds --prefix $mt --matrix_ml $MATRIX
            echo "Done"
            echo
            echo
            echo
        fi
    done
done

# cli="run.py '--dataset', 'electricity', '--prefix', 'avg', '--percent_mc', '01', '--missing_mode', 'noadjacency' '--weight-mechanism'"

# --dataset electricity --prefix avg --percent_mc 01 --missing_mode noadjacency --weight-mechanism 0