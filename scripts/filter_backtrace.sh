#!/bin/bash

usage() {
    echo "Usage: $0 -i INPUT -d DURATION -u INTERVAL"
    exit 1
}

# 解析命令行参数
while getopts "i:d:u:" opt; do
    case $opt in
        i)
            INPUT=$OPTARG
            ;;
        d)
            DURATION=$OPTARG
            ;;
        u)
            INTERVAL=$OPTARG
            ;;
        *)
            usage
            ;;
    esac
done

# 检查必要的参数是否已提供
if [ -z "$INPUT" ] || [ -z "$DURATION" ] || [ -z "$INTERVAL" ]; then
    usage
fi


TRACE_DIR="/path/to/MPI_profile/${INPUT}"
STAD_DIR="/path/to/STAD"

NAME="${INPUT}_${DURATION}ms_closed"
FILTER_DIR="${STAD_DIR}/result/backtrace"

echo "Running analysis with the following parameters:"
echo "Trace directory: ${TRACE_DIR}"
echo "Output directory: ${FILTER_DIR}"
echo "Duration: ${DURATION}ms"
echo "Interval: ${INTERVAL}ms"

echo "Duration: $(( DURATION * 2500000 ))"
echo "Interval: $(( INTERVAL * 2500000 ))"

timeslice_analysis -i ${TRACE_DIR}/trace -d ${TRACE_DIR}/lineinfo -s ${FILTER_DIR}/${NAME}_abnormal_indices.txt -f -o ${FILTER_DIR}/${NAME}_backtrace -l $(( INTERVAL * 2500000 )) -u $(( DURATION * 2500000 )) -b
