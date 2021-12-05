#!/usr/bin/env bash

sub_dir=results
for smooth in {1..10}; do
    smooth_factor=`echo $smooth|awk '{print $1/10}'`
    for r in {0..2}; do
        for s in {0..3}; do
            shift=`expr ${r} \* 4 + ${s}`
            if [[ $shift -gt 9 ]];then
                break
            fi
            output_file_name="./${sub_dir}/precision_${smooth}_${shift}"
            output_file="./${sub_dir}/${output_file_name}.json"
            log_file="${output_file_name}.log"
            echo "smooth factor: $smooth_factor, shift: $shift"
            echo "./src/test.py --dir ./20_newsgroups --strategy alternative --direction asc --training-percent 50 --shift ${shift} --term-threshold 0 --stop-words ./src/stop_words-long.txt --smooth-factor ${smooth_factor} --output ${output_file}"
            nohup ./src/test.py --dir ./20_newsgroups --strategy alternative --direction asc --training-percent 50 --shift ${shift} --term-threshold 0 --stop-words ./src/stop_words-long.txt --smooth-factor ${smooth_factor} --output ${output_file} > ${log_file} 2>&1 &
        done
        sleep 300
    done
    echo ""
done


