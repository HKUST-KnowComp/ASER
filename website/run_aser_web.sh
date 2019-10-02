#!/usr/bin/env bash
export FLASK_APP=aserweb
export FLASK_ENV=development
export ASER_WEB_PORT=$1
export ASER_HOST=$2
export ASER_PORT=$3
export ASER_PORT_OUT=$4
flask run --host=0.0.0.0 --port $1
