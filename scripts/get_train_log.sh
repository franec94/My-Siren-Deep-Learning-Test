#!/usr/bin/env bash

user="fake-user"
remote_server="fake.server"
dest_filename="fake.dest.filename"

src_path="fake/src/path"
src_filename="fake.src.filename"

CMD="pscp"

${CMD} ${user}@${remote_server}:${src_path}/${src_filename} ${dest_filename}

if [ $? != 0 ] ; then
  echo "Error: ${CMD} ${user}@${remote_server}:${src_path}/${src_filename} ${dest_filename}"
  exit -1
fi

exit 0
