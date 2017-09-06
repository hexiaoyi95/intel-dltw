#!/bin/bash

get_idle_partition() {
  ptions=$(sinfo | grep ${1}.*idle | cut -d' ' -f1 )
  ptionArray=(${ptions// /})
  for p in "${ptionArray[@]}";do
    if [[ "${p: -1}" == "*" ]]; then
      p="${p::-1}"
    fi
    if [[ $p != ${1}-fast ]];then
      ption=${p}
      break
    fi
  done
  if [[ $ption == "" ]];then
    if [[ $2 -eq 1 ]];then
      ption=${1}-inf
    else
      ption=${1}-debug
    fi
  fi
  echo $ption
}

cd ~/public-dltw
rm -rf ./logs
cur_date=$(date +%Y-%m-%d_%H_%M_%S)
nws=${WORKSPACE}/${cur_date}

if [[ ${first_build} == "true" ]];then
  previousDir=$nws
  previousDate=$cur_data
else
  previousDir=$(ls -td ${WORKSPACE}/*/ | head -1)
  previousData="$(basename $previousDir)"
fi

echo $previousDir

mkdir -p ${nws}/

mkdir ./logs

mapfile -t targets <<< "${cpu}"
mapfile -t cases <<< "${case}"
CAFFE_HOME=${caffe_home}
CAFFE_COMMIT_ID=${caffe_commit_id}
if [[ $CAFFE_HOME == "" ]];then
  CAFFE_HOME="dl-frameworks/dl_framework-intel_caffe"
  rm -rf $CAFFE_HOME
  mkdir $CAFFE_HOME
  git clone ssh://daisyden@git-ccr-1.devtools.intel.com:29418/dl_framework-intel_caffe $CAFFE_HOME > logs/build_caffe.log 2>&1
  cd $CAFFE_HOME
  if [[ $CAFFE_COMMIT_ID != "" ]];then
    git checkout $CAFFE_COMMIT_ID
  fi
  rm -rf build
  mkdir build
  cd build
  cmake .. -DCPU_ONLY=ON
  make -j$(nproc)
  cd ~/public-dltw
fi


declare -A pids
#declare -A isRef

if [[ ${rerun_ref} == "on" ]];then
  for target in "${targets[@]}"; do
    for case_ in "${cases[@]}"; do
      partition=$(get_idle_partition ${target})
      case_name=${case_##*/}
      case_name=${case_name%%.*}

      srun -p ${partition} ./bin/run_suite.py -pp $CAFFE_HOME/python -p "test"-${target}-${case_name} -c ${case_} -r "on" -cpu $target > ./logs/${target}-${case_name}-"ref".log  2>&1 &
      echo "submitted job test-${target}-${case_name}-ref to partition ${partition}"
      pids["test-${target}-${case_name}-ref"]="$!"
      #isRef["test-${target}-${case_name}-ref"]="1"

    done
  done

  echo "waiting for running reference"
  while (( ${#pids[@]}  > 0 )) ; do
    for cur_job in "${!pids[@]}"; do
      if ! kill -0 "${pids[$cur_job]}" 2>/dev/null ; then
        if wait "${pids[$cur_job]}" ; then
          echo "$cur_job finished and successed"
        else
          echo "$cur_job finished but failed"
        fi
        unset pids[$cur_job]
      fi
    done
    sleep 1
  done
  echo "re-generating reference finished"
fi

for target in "${targets[@]}"; do
  for case_ in "${cases[@]}"; do
    partition=$(get_idle_partition ${target})
    case_name=${case_##*/}
    case_name=${case_name%%.*}
    srun -p ${partition} ./bin/run_suite.py -pp $CAFFE_HOME/python -p "test"-${target}-${case_name} -c ${case_} -r "off" -cpu $target > ./logs/${target}-${case_name}.log  2>&1 &
    echo "submitted job test-${target}-${case_name} to partition ${partition}"
    pids["test-${target}-${case_name}"]="$!"
    #isRef["test-${target}-${case_name}"]="0"
  done
done

echo "waiting for submitted jobs"
result_json=""
while (( ${#pids[@]}  > 0 )) ; do
  for cur_job in "${!pids[@]}"; do
    if ! kill -0 "${pids[$cur_job]}" 2>/dev/null ; then
      if wait "${pids[$cur_job]}" ; then
        echo "$cur_job finished and successed"
        #if [[ ${isRef[$cur_job]} == "0" ]];then
          cp -r $cur_job/test_report ${nws}/$cur_job
          cp $cur_job/test_results.json ${nws}/$cur_job
          result_json="$result_json $cur_job/test_results.json"
        #fi
      else
        echo "$cur_job finished but failed"
      fi
      unset pids[$cur_job]
    fi
  done
  sleep 1
done

rm generate_html/test_result.html
python generate_html/test_jinja.py ${JOB_URL}/ws/${cur_date} ${JOB_URL}/ws/$previousDate ${previousDir} $result_json
cp generate_html/test_result.html ${nws}
cp generate_html/test_result.html ${WORKSPACE}

echo "finished"

