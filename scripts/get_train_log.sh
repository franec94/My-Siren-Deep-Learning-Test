#!/usr/bin/env bash

# --- Global Variables ---
script_name=$0
src_path=""
declare -A conf_table


# --- Functions ---
function update_src_path() {
    keys_for_src_path=("user" "date" "timestamp")

    src_path=${conf_table["src_path"]}
    for k in ${keys_for_src_path[@]} ; do
      # echo $k
      val=${conf_table["${k}"]}
      src_path="${src_path/${k}/${conf_table["${k}"]}}" 
    done
    # echo $src_path
    
}

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

# --- Start Script ---
check_cmd_args $@

conf_file=$1
create_conf_array ${conf_file}

update_src_path

# src_path="${conf_table["src_path"]/user/${conf_table["user"]}}" 
# src_path="${src_path/date/${conf_table["date"]}}"
# src_path="${src_path/timestamp/${conf_table["timestamp"]}}"
# echo ${src_path}

CMD="pscp"
echo "${CMD} ${conf_table["user"]}@${conf_table["remote_server"]}:${src_path}/${conf_table["src_filename"]} ${conf_table["dest_filename"]}"
# ${CMD} ${conf_table["user"]}@${conf_table["remote_server"]}:${src_path}/${conf_table["src_filename"]} ${conf_table["dest_filename"]}

# cat ${conf_table["dest_filename"]} | sed 's/INFO:root://g' | grep -E "^hidden_layers" | sort | uniq -c

exit 0

