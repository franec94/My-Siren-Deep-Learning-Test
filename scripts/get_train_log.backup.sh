#!/usr/bin/env bash

script_name=$0

declare -A conf_table


function create_conf_array() {
    local conf_file=$1

    while IFS= read -r line ; do
      # echo "$line"
      if [ ${#line} == 0 ] ; then
        continue
      fi

      key=$(echo ${line} | cut -d "=" -f 1)
      value=$(echo ${line} | cut -d "=" -f 2)
      # echo "k=${key} v=${value}"

      conf_table["${key}"]="${value}"
      # echo "${conf_table[${key}]}"

    done < "$conf_file"

    unset conf_file
}

function check_cmd_args() {
    local cmd_args=$@

    if [ ${#cmd_args[@]} != "1" ] ; then
      echo "${#cmd_args[@]}"
      echo "Usage: $script_name conf.txt"
      exit -1
    elif [ ! -f ${cmd_args[0]} ] ; then
      echo "Error: '${cmd_args[0]}' is not a file!"
      exit -1
    fi
    unset cmd_args
}


check_cmd_args $@

conf_file=$1
create_conf_array ${conf_file}

CMD="pscp"
# ${CMD} ${conf_table["user"]}@${conf_table["remote_server"]}:${conf_table["src_path"]}/${conf_table["src_filename"]} ${conf_table["dest_filename"]}
echo "${CMD} ${conf_table["user"]}@${conf_table["remote_server"]}:${conf_table["src_path"]}/${conf_table["src_filename"]} ${conf_table["dest_filename"]}"


exit 0

user="chiarlo"
remote_server="iside.polito.it"
dest_filename="train_log.txt"

src_path="/home/chiarlo/siren-project/results/23-10-2020/1603410154-248962/train"
src_filename="train.log"

CMD="pscp"
${CMD} ${user}@${remote_server}:${src_path}/${src_filename} ${dest_filename}

# CMD="ssh"
# ssh ${user}@${remote_server} 'tail -f ${src_path}/${src_filename}' > ${dest_filename}


cat ${dest_filename} | sed 's/INFO:root://g' | grep -E "^hidden_layers" | sort | uniq -c

exit 0
